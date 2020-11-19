from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pathlib
from skimage import io, transform
from PIL import Image


def get_image_dataset(dataroot, config):
    # Create the dataset
    dataset = datasets.ImageFolder(root=dataroot,
                                   transform=create_transform(config))
    return dataset


def create_transform(config):
    return transforms.Compose([
        transforms.Resize(config.imageSize),
        transforms.CenterCrop(config.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class InpaintDataset(Dataset):
    def __init__(self, directory, image_size, mask_generator):
        self.directory = pathlib.Path(directory)
        self.images_filename = list(self.directory.glob('*.jpg'))
        self.image_size = image_size
        self.mask_generator = mask_generator

    def __apply_mask(self, image, mask):
        image_copy = np.copy(image)
        image_copy[mask == 0] = 1

        return image_copy

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        target_image = io.imread(self.images_filename[idx])
        target_image = transform.resize(target_image, self.image_size)
        mask = self.mask_generator(target_image)
        corrupted_image = self.__apply_mask(target_image, mask)

        return target_image, corrupted_image, mask
