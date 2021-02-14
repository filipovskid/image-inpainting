from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pathlib

from inpaint_model import InpaintModel
from DCGAN.DCGAN_config import _C as _dcganConfig
from styleGAN.styleGAN_config import _C as _styleganConfig
from datasets.datasets import InpaintDataset

import torch
import torchvision
from torchvision import utils
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt-path', help='Path to ground truth data', type=pathlib.Path, dest='gt_path')
    parser.add_argument('--masks-path', help='Path to mask data', type=pathlib.Path, dest='masks_path')
    parser.add_argument('--image-size', help='Size to transform input images and masks to', type=int, dest='image_size')
    parser.add_argument('--batch-size',
                        help='Size of inpainting batch. If 0, entire dataset is processed as a single batch',
                        type=int,
                        default=0,
                        dest='batch_size')
    parser.add_argument('--type', choices=['dcgan', 'stylegan'], help='GAN model')
    parser.add_argument('--checkpoint', help="Path to model's checkpoint", type=pathlib.Path)
    parser.add_argument('--output-path', help='Path to output data', type=pathlib.Path, dest='output_path')
    args = parser.parse_args()

    return args


def get_transforms(image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    return transform, mask_transform


def dcgan_inpaint_model(checkpoint):
    return InpaintModel(model_filename=checkpoint, config=_dcganConfig, gan_type='dcgan')


def stylegan_inpaint_model(checkpoint):
    return InpaintModel(model_filename=checkpoint, config=_styleganConfig, gan_type='stylegan')


def tensor_from_ndarray(image_batch):
    return torch.from_numpy(image_batch).permute(0, 3, 1, 2)


def save_evaluation(image_dicts, names, output_path):
    images_data = []

    for data in image_dicts:
        if type(data['images']) == np.ndarray:
            images_data.append({'name': data['name'], 'images':tensor_from_ndarray(data['images'])})
        else:
            images_data.append(data)

    for i, name in enumerate(names):
        save_path = output_path.joinpath(name.split('.')[0])
        save_path.mkdir(parents=True, exist_ok=True)

        for data in images_data:
            utils.save_image(data['images'][i].float(), save_path.joinpath(f'{data["name"]}-{name}'))


def inpaint_images(gt_path, masks_path, image_size, batch_size, output_path, gt_output_path, inpaint_output_path,
                   model_type, checkpoint):
    inpaint_models = {
        'dcgan': lambda chkp: dcgan_inpaint_model(chkp),
        'stylegan': lambda chkp: stylegan_inpaint_model(chkp)
    }
    transform, mask_transform = get_transforms(image_size)

    inpainter = inpaint_models[model_type](checkpoint)
    inpaint_dataset = InpaintDataset(gt_path, masks_path, transform=transform, mask_transform=mask_transform)
    batch_size = len(inpaint_dataset) if batch_size == 0 else batch_size
    dataloader = DataLoader(inpaint_dataset, batch_size=batch_size)

    for target_images, corrupted_images, masks, names in dataloader:
        importance_weights, G_z, inpainted_images = inpainter.inpaint(corrupted_images=corrupted_images,
                                                                    image_masks=masks)
        image_dicts = [
            {'name': 'target_image', 'images': target_images},
            {'name': 'corrupted_image', 'images': corrupted_images},
            {'name': 'mask', 'images': masks},
            {'name': 'importance_weight', 'images': importance_weights},
            {'name': 'G_z', 'images': G_z},
            {'name': 'inpainted_image', 'images': inpainted_images},
        ]

        save_evaluation(image_dicts, names=names, output_path=output_path)

        inpainted_images = torch.from_numpy(inpainted_images).permute(0, 3, 1, 2)
        for i, name in enumerate(names):
            utils.save_image(target_images[i], gt_output_path.joinpath(name))
            utils.save_image(inpainted_images[i], inpaint_output_path.joinpath(name))


def main():
    args = parse_args()
    output_path = args.output_path
    gt_output_path = output_path.joinpath('ground_truth')
    inpainted_output_path = output_path.joinpath('inpainted')

    output_path.mkdir(parents=True, exist_ok=True)
    gt_output_path.mkdir(parents=True, exist_ok=True)
    inpainted_output_path.mkdir(parents=True, exist_ok=True)

    inpaint_images(gt_path=args.gt_path,
                   masks_path=args.masks_path,
                   image_size=args.image_size,
                   batch_size=args.batch_size,
                   output_path=output_path,
                   gt_output_path=gt_output_path,
                   inpaint_output_path=inpainted_output_path,
                   model_type=args.type,
                   checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
