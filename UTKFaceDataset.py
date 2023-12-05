import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class UTKFaceDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.file_list = os.listdir(self.dir_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.dir_path, img_name)
        image = Image.open(img_path)
        label = self.parse_filename(img_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def parse_filename(self, filename):
        filename = filename.split('_')
        age, gender, race = map(int, filename[:3])
        return {'age': age, 'gender': gender, 'race': race}

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

# Dataset for a dataset balanced on age, gender, race
class BalancedDataset(Dataset):
    def __init__(self, csv_file, sample_weights=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.weights = sample_weights

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_item = self.df.iloc[idx]
        image = Image.open(sample_item['file_path'])
        label = {'age': sample_item['age'], 'gender': sample_item['gender'], 'race': sample_item['race'], 'file': sample_item['file_path']}
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        elif self.transform is None:
            sample['image'] = ToTensor(sample['image'])

        return sample