"""
PTC Whole Slide
Contains all functions for transforming and loading data.
"""

from pathlib import Path
from typing import (Dict, Tuple, List)
import numpy as np
import torch
from torch import manual_seed
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

###########################################
#           TRANSFORMING DATA             #
###########################################
def get_data_transforms(resize: int,
                        center_crop: int,
                        color_jitter_brightness: float,
                        color_jitter_contrast: float,
                        color_jitter_saturation: float,
                        color_jitter_hue: float,
                        ) -> Dict[str, transforms.Compose]:
    """
    Sets up the dataset transforms for training and validation.
    Args:
        resize: Resize the input PIL Image to the given size.
        center_crop: Desired output size of the crop.
        color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
    Returns:
        A dictionary mapping training and validation strings to data transforms.
    """
    return {
        "train":
        transforms.Compose(transforms=[
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ColorJitter(brightness=color_jitter_brightness,
                                   contrast=color_jitter_contrast,
                                   saturation=color_jitter_saturation,
                                   hue=color_jitter_hue),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val":
        transforms.Compose(transforms=[
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test":
        transforms.Compose(transforms=[
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        }

###########################################
#               LOADING DATA              #
###########################################
def load_data(image_folder: Path,
              batch_size: int,
              resize: int,
              center_crop: int,
              color_jitter_brightness: float,
              color_jitter_contrast: float,
              color_jitter_saturation: float,
              color_jitter_hue: float,
              include_test: bool,
              ) -> Tuple[Dict[str, torch.utils.data.DataLoader], Dict[str, int]]:
    """
    Transform and load the training and validation data
    Args:
        image_folder: Location of the folder containing training and validation datasets.
        resize: Resize the input PIL Image to the given size.
        center_crop: Desired output size of the crop.
        color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
        include_test:

    Returns:
        A dictionary mapping training and validation strings to DataLoader.
        A dictionary mapping the number to categories.

    """


    manual_seed(0)

    if include_test:
        data_folders = ('train', 'val', 'test')
    else:
        data_folders = ('train', 'val')

    data_transforms = get_data_transforms(
        resize=resize,
        center_crop=center_crop,
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        color_jitter_saturation=color_jitter_saturation,
        color_jitter_hue=color_jitter_hue
        )

    image_datasets = {
        x: datasets.ImageFolder(root=str(image_folder.joinpath(x)),
                                transform=data_transforms[x])
        for x in data_folders}


    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True)
        for x in data_folders}


    dataset_sizes = {
        x: len(image_datasets[x])
        for x in data_folders}

    classes = image_datasets['train'].classes

    print(
        f"\n{dataset_sizes['train']} train patches"
        f"\n{dataset_sizes['val']} validation patches")

    if include_test:
        print(
            f"\n{dataset_sizes['test']} test patches"
        )

    return dataloaders, classes

###########################################
#          VISUALAZING IMAGE PATCHES      #
###########################################

#show multiple images
def multishow(dataloaders: Dict[str, torch.utils.data.DataLoader],
              show_folder: str, num_per_class: int, title: str, classes: List[str]) -> None:
    """
    Show multiple image patches in per classes
    Args:
        dataloaders: A dictionary mapping DataLoader to train, val and test
        show_folder: The image folder we want to show (train, val, test)
        num_per_class: Number of images of each class want to display
        title: The title for the display Images
        classes: Name of each class
    """


    images, labels = next(iter(dataloaders[show_folder]))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, ax = plt.subplots(nrows=2, ncols=num_per_class, figsize=(10, 8))

    fig.suptitle(t=title, fontsize=20)

    for i in range(num_per_class):
        image_postive = images[labels == 1]
        image = image_postive[i].numpy().transpose((1, 2, 0))
        image = image * std + mean
        image = np.clip(image, 0, 1)
        ax[0, i].imshow(image)
    ax[0, 0].set_ylabel(ylabel=classes[0], size='large')

    for i in range(num_per_class):
        image_negative = images[labels == 0]
        image = image_negative[i].numpy().transpose((1, 2, 0))
        image = image * std + mean
        image = np.clip(image, 0, 1)
        ax[1, i].imshow(image)
    ax[1, 0].set_ylabel(ylabel=classes[1], size='large')
