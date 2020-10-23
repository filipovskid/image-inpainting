from torchvision import datasets
from torchvision import transforms


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
