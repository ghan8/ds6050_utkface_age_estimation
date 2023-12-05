from UTKFaceDataset import *
from UTKFaceModelTrainer import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from tqdm import tqdm
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import random

def count_files(folder_path):
    """
    Counts the number of files in a given folder path.

    Parameters:
    folder_path (str): The path to the folder.

    Returns:
    int: The number of files in the folder.
    """
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

def remove_files(folder_path, file_type):
    """
    Removes files of a specific file type from a given folder path.

    Parameters:
    folder_path (str): The path to the folder.
    file_type (str): The file type to be removed.

    Returns:
    None
    """
    for file in os.listdir(folder_path):
        if file.endswith(file_type):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            
def move_files(source_directory, destination_directory, num_files):
    """
    Moves a specified number of files from a source directory to a destination directory.

    Parameters:
    source_directory (str): The path to the source directory.
    destination_directory (str): The path to the destination directory.
    num_files (int): The number of files to be moved.

    Returns:
    None
    """
    # Get the list of files in the source directory
    file_list = os.listdir(source_directory)
    
    # Randomly select the specified number of files to move
    files_to_move = random.sample(file_list, num_files)
    
    # Move the selected files to the destination directory
    for file_name in files_to_move:
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(destination_directory, file_name)
        os.rename(source_path, destination_path)

def get_gender_label(gender):
    """
    Returns the label for a given gender.

    Parameters:
    gender (int): The gender value.

    Returns:
    str: The label for the gender.
    """
    gender_map = {0: 'Male', 1: 'Female'}
    return gender_map[gender.item()]

def get_race_label(race):
    """
    Returns the label for a given race.

    Parameters:
    race (int): The race value.

    Returns:
    str: The label for the race.
    """
    race_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
    return race_map[race.item()]

def calculate_normalization_mean_std(data_loader):
    """
    Calculates the mean and standard deviation for normalization from a given data loader.

    Parameters:
    data_loader (DataLoader): The data loader.

    Returns:
    tuple: A tuple containing the mean and standard deviation.
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for batch in data_loader:
        images = batch['image']
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_size

    mean /= total_samples
    std /= total_samples

    return mean, std

def display_image_file_path(image_file_path):
    """
    Displays an image given its file path.

    Parameters:
    image_file_path (str): The file path of the image.

    Returns:
    None
    """
    image = Image.open(image_file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def display_image_df(df, file_path):
    """
    Displays an image given its file path and additional information from a dataframe.

    Parameters:
    df (DataFrame): The dataframe containing the image information.
    file_path (str): The file path of the image.

    Returns:
    None
    """
    # look up the file path in the dataframe, use the filepath key to identify the age, gender, race 
    # use the row to identify the age
    age = df[df['file_path'] == file_path]['age'].values[0]
    gender = df[df['file_path'] == file_path]['gender'].values[0]
    race = df[df['file_path'] == file_path]['race'].values[0]

    # use the age, gender, and age to label and display the image
    image = Image.open(file_path)
    plt.imshow(image)
    plt.title(f"Age: {age}, Gender: {get_gender_label(gender)}, Race: {get_race_label(race)}")
    plt.axis('off')
    plt.show()

def display_image(image):
    """
    Displays an image.

    Parameters:
    image (PIL.Image.Image): The image to be displayed.

    Returns:
    None
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_random_images(df, num_images):
    """
    Displays a specified number of random images from a dataframe.

    Parameters:
    df (DataFrame): The dataframe containing the image information.
    num_images (int): The number of random images to be displayed.

    Returns:
    None
    """
    random_indices = random.sample(range(len(df)), num_images)
    images = [Image.open(df['file_path'][i]) for i in random_indices]
    labels = [f"Age: {df['age'][i]}, Gender: {get_gender_label(df['gender'][i])}, Race: {get_race_label(df['race'][i])}" for i in random_indices]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(labels[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
