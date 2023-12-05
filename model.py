import torch
import torch.nn as nn
import torchvision.models as models 
from torchvision.models import vision_transformer, vit_b_16
import os

class Model(nn.Module):
    def __init__(self, model_name, version, num_classes, pretrained=True):
        super(Model, self).__init__()
        
        self.model_name = model_name  # Store the model name as an attribute
        
        if pretrained == True:
            transfer = "pretrained"
        else:
            transfer = "scratch"
        
        if model_name == 'vgg16':
            # Load the pretrained vgg16 model
            vgg16 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=pretrained)
            # Freeze training for all layers
            for param in vgg16.features.parameters(): 
                param.require_grad = False
            # Newly created modules have require_grad=True by default
            num_features = vgg16.classifier[6].in_features
            # Remove last layer
            features = list(vgg16.classifier.children())[:-1]
            # Add our layer with num classes
            features.extend([nn.Linear(num_features, num_classes)])
            # Replace the model classifier
            vgg16.classifier = nn.Sequential(*features)
			
            # construct the model file name to load trained weights
            model_file_name = f"{model_name}_{transfer}_{version}.pth"
            #if os.path.exists(model_file_name):
            #    vgg16.load_state_dict(torch.load(model_file_name))
            
            self.model = vgg16
            
        elif model_name == 'resnet50':
            # Load the pretrained resnet50 model
            resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=pretrained)
            # Freeze training for all layers except the last one
            for param in resnet50.parameters():
                param.requires_grad = False
            # Newly created modules have require_grad=True by default
            # Replace the last layer with a new layer
            resnet50.fc = nn.Sequential(
                nn.Linear(resnet50.fc.in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(4096, num_classes)
            )
            # construct the model file name to load trained weights
            model_file_name = f"{model_name}_{transfer}_{version}.pth"
            #if os.path.exists(model_file_name):
            #    resnet50.load_state_dict(torch.load(model_file_name))
            self.model = resnet50
            
        elif model_name == 'vitb16':
            if pretrained == True:
            	vitb16 = models.vit_b_16(weights='IMAGENET1K_SWAG_LINEAR_V1')
            else: 
                vitb16 = models.vit_b_16()
                
            # Freeze training for all layers except the last one
            for param in vitb16.parameters():
                param.requires_grad = False
            
            # Extend the model.head with another layer for 116 class output classification
            vitb16.heads = nn.Sequential(
                 vitb16.heads.head,
                 nn.Linear(in_features=vitb16.heads.head.out_features, out_features=4096),
                 nn.ReLU(inplace=True),
                 nn.Dropout(p=0.5, inplace=False),
                 nn.Linear(4096, 4096),
                 nn.ReLU(inplace=True),
                 nn.Dropout(p=0.5, inplace=False),
                 nn.Linear(4096, num_classes)
             )

            # construct the model file name to load trained weights
            model_file_name = f"{model_name}_{transfer}_{version}.pth"
            #if os.path.exists(model_file_name):
            #    vitb16.load_state_dict(torch.load(model_file_name))
                
            self.model = vitb16
        
        else:
            raise ValueError("Unsupported model name. Choose from 'vgg16', 'resnet50', '....'.")
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
