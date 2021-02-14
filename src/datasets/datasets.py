from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pathlib
# from skimage import io, transform
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


def list_files(dir_path):
    p = dir_path.glob('**/*')
    return [x for x in p if x.is_file()]


class InpaintDataset(Dataset):
    def __init__(self, gt_path, masks_path, transform, mask_transform=None):
        self.gt_path = pathlib.Path(gt_path)
        self.masks_path = pathlib.Path(masks_path)
        self.gt_image_filenames = sorted(list_files(self.gt_path))
        self.mask_filenames = sorted(list_files(self.masks_path))

        min_files = min(len(self.gt_image_filenames), len(self.mask_filenames))
        assert(min_files > 0)

        self.gt_image_filenames = self.gt_image_filenames[:min_files]
        self.mask_filenames = self.mask_filenames[:min_files]
        self.transform = transform
        self.mask_transform = mask_transform

    def __create_2channel_mask(self, mask):
        if len(mask.shape) == 2:
            return mask

        assert (len(mask.shape) == 3)
        return mask[:, :, 0]

    def __binarize_masks(self, mask):
        bmask = np.empty_like(mask)
        bmask[mask > 0] = 1
        bmask[mask <= 0] = 0

        return bmask

    def __apply_mask(self, image, mask):
        image_copy = np.copy(image)
        image_copy[:, mask == 0] = 1

        return image_copy

    def __len__(self):
        return len(self.gt_image_filenames)

    def __getitem__(self, idx):
        name = self.gt_image_filenames[idx].name
        target_image = Image.open(self.gt_image_filenames[idx])
        target_image = self.transform(target_image)

        if self.mask_transform:
            mask = self.mask_transform(Image.open(self.mask_filenames[idx]))
        else:
            mask = Image.open(self.mask_filenames[idx])

        mask = np.asarray(mask)
        mask = self.__binarize_masks(mask)
        mask = self.__create_2channel_mask(mask)
        corrupted_image = self.__apply_mask(target_image, mask)

        return target_image, corrupted_image, mask, name
