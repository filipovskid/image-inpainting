from torchvision import datasets
from torchvision import transforms
from models.DCGAN_config import _C


def get_image_dataset(dataroot):
    # Create the dataset
    dataset = datasets.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                   transforms.Resize(_C.imageSize),
                                   transforms.CenterCrop(_C.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    return dataset
