from torchvision import datasets
from torch.utils import data
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = Path(self.imgs[index][0])
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def prepare(in_path, out_path, size):
    data_transforms = transforms.Compose([
        transforms.Resize(size)
    ])

    dataset = ImageFolderWithPaths(in_path, transform=data_transforms)

    for image, _, image_path in dataset:
        output_path = out_path.joinpath(image_path.relative_to(in_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, help='Path to the dataset')
    parser.add_argument('-o', '--output', type=Path, help='Path to output directory')
    parser.add_argument('--size', type=int, help='Size of the smaller side of the image')

    args = parser.parse_args()
    prepare(args.input, args.output, args.size)
