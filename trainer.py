!pip install torchvision
from UTKFaceDataset import *
from UTKFaceModelTrainer import *
from UTKFaceUtils import *
from model import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from tqdm import tqdm
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
!pip freeze > requirements.txt

# Define the transformation pipeline
# https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/#
# https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(degrees=10),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5904, 0.4462, 0.3769], std=[0.2177, 0.1899, 0.1799])
])

batch_size = 32

# balanced dir_path, transformed for vgg
train_dataset = UTKFaceDataset(dir_path='UTKFACE', transform=data_transforms)
balanced_dataset = BalancedDataset('combined_data.csv', transform=data_transforms)
val_dataset = UTKFaceDataset(dir_path='UTKFACE_val', transform=data_transforms)

# Create data loaders for train and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
balanced_loader = torch.utils.data.DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Create a random sampler for the train dataset
# set to a small number for troubleshooting
train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=10000)
balanced_sampler = torch.utils.data.RandomSampler(balanced_dataset, replacement=True, num_samples=10000)
val_sampler = torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=1000)

# Create a data loader for the train dataset with the random sampler
train_sample_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
balanced_sample_loader = torch.utils.data.DataLoader(balanced_dataset, batch_size=batch_size, sampler=balanced_sampler)
val_sample_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)


model = Model(model_name='vgg16', version = 7, num_classes=116, pretrained=True)
model_name = "vgg16_pretrained_7.pth"

for param in model.parameters():
        param.requires_grad = True
  
# load model weights if starting from a tuned version
# model.load_state_dict(torch.load('vgg16_pretrained_1.pth'))

# specify device and send the model to the device
device = torch.device("cpu")
model = model.to(device)

num_epochs = 100
patience = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

mae_list = []
accuracy_list = []  
epoch_list = []
training_loss_list = []
validation_loss_list = []
lowest_mae = float('inf')
highest_accuracy = -float('inf')
no_improve = 0
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    training_loss_loop = []
    
    for i in tqdm(range(len(train_loader)), desc='Training'):
        try:
            next_element = next(iter(train_loader))
        except:
            continue
        images = next_element['image'].to(device)
        labels = next_element['label']['age'].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        training_loss_loop.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    training_loss_avg = sum(training_loss_loop) / len(training_loss_loop)
    training_loss_list.append(training_loss_avg)

    model.eval()
    total_samples = 0
    total_error = 0
    correct_predictions = 0  
    validation_loss_loop = []
    with torch.no_grad():
        for i in tqdm(range(len(val_loader)), desc='Validation'):
            try:
                next_element = next(iter(val_loader))
            except:
                continue
            images = next_element['image'].to(device)
            labels = next_element['label']['age'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  
            softmax_outputs = torch.softmax(outputs, dim=1)
            max_prob_element = torch.argmax(softmax_outputs, dim=1)
            validation_loss_loop.append(loss.item())
            total_samples += labels.size(0)
            total_error += F.l1_loss(max_prob_element.float(), labels.float(), reduction='sum').item() 
            correct_predictions += (max_prob_element == labels).sum().item()  

    validation_loss_avg = sum(validation_loss_loop) / len(validation_loss_loop)
    validation_loss_list.append(validation_loss_avg)

    mae = total_error / total_samples
    mae_list.append(mae)
    accuracy = correct_predictions / total_samples  
    accuracy_list.append(accuracy)  
    epoch_list.append(epoch)

    if mae < lowest_mae:
        lowest_mae = mae
        torch.save(model.state_dict(), model_name)
        no_improve = 0
    else:
        no_improve += 1
    
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        torch.save(model.state_dict(), model_name)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss_avg:.4f}, Validation Loss: {validation_loss_avg:.4f}, MAE: {mae:.4f}, Accuracy: {accuracy:.4f}')  

    scheduler.step(validation_loss_avg)

    if no_improve >= patience:
        print('Early stopping...')
        break

# Save model performance across training epochs
model_performance = {
    'mae_list': mae_list,
    'accuracy_list': accuracy_list,
    'epoch_list': epoch_list,
    'training_loss_list': training_loss_list,
    'validation_loss_list': validation_loss_list
}

model_name = model_name.rstrip(".pth")
pd.DataFrame(model_performance).to_csv(f'{model_name}_model_performance.csv')