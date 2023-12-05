from UTKFaceDataset import *
from UTKFaceModelTrainer import *
from UTKFaceUtils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self, model, model_name, train_loader, val_loader, device='cpu', num_epochs=100, patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.patience = patience
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7, weight_decay=0.00001)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = model_name
        self.no_improve = 0
        self.lowest_mae = float('inf')
        self.highest_val_accuracy = -float('inf')
        self.metric_logger = {'train_loss': [], 'val_loss': [], 'mae': [], 'accuracy': []}

    def train_epoch(self):
        self.model.train()
        training_loss_loop = []
        for i in tqdm(range(len(self.train_loader)), desc='Training'):
            try:
                next_element = next(iter(self.train_loader))
            except:
                continue
            images = next_element['image'].to(self.device)
            labels = next_element['label']['age'].to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            training_loss_loop.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        avg_loss = sum(training_loss_loop) / len(training_loss_loop)
        self.metric_logger['train_loss'].append(avg_loss)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        total_samples = 0
        total_error = 0
        correct_predictions = 0
        validation_loss_loop = []
        with torch.no_grad():
            for i in tqdm(range(len(self.val_loader)), desc='Validation'):
                try:
                    next_element = next(iter(self.val_loader))
                except:
                    continue
                images = next_element['image'].to(self.device)
                labels = next_element['label']['age'].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                softmax_outputs = torch.softmax(outputs, dim=1)
                max_prob_element = torch.argmax(softmax_outputs, dim=1)
                validation_loss_loop.append(loss.item())
                total_samples += labels.size(0)
                total_error += F.l1_loss(max_prob_element.float(), labels.float(), reduction='sum').item()
                correct_predictions += (max_prob_element == labels).sum().item()
        avg_loss = sum(validation_loss_loop) / len(validation_loss_loop)
        mae = total_error / total_samples
        accuracy = correct_predictions / total_samples
        self.metric_logger['val_loss'].append(avg_loss)
        self.metric_logger['mae'].append(mae)
        self.metric_logger['accuracy'].append(accuracy)
        return avg_loss, mae, accuracy

    def trainer(self):
        for epoch in range(self.num_epochs):
            training_loss_avg = self.train_epoch()
            validation_loss_avg, mae, accuracy = self.validate_epoch()
            if mae < self.lowest_mae:
                self.lowest_mae = mae
                torch.save(self.model.state_dict(), self.model_name)
                self.no_improve = 0
            else:
                self.no_improve += 1
            if accuracy > self.highest_val_accuracy:
                self.highest_val_accuracy = accuracy
                torch.save(self.model.state_dict(), self.model_name)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {training_loss_avg:.4f}, Validation Loss: {validation_loss_avg:.4f}, Val MAE: {mae:.4f}, Val Accuracy: {accuracy:.4f}')
            self.scheduler.step(validation_loss_avg)
            if self.no_improve >= self.patience:
                print('Early stopping...')
                break
        return self.metric_logger