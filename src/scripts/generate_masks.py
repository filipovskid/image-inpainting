from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ..mask_generator import MaskGenerator
from PIL import Image
import numpy as np
import pathlib


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-path', dest='output_path', help='Path to output data', type=pathlib.Path)
    parser.add_argument('--type', choices=['rr', 'cr', 'noise'])
    parser.add_argument('--lower', type=float, default=0.1, help='Mask percent lower bound')
    parser.add_argument('--upper', type=float, default=0.4, help='Mask percent upper bound')
    parser.add_argument('-p', '--percent', type=float, default=0, help='Mask noise percent')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--count', type=int, help='Number of masks to be created')
    parser.add_argument('--seed', type=int, default=42, help='Seed')

    args = parser.parse_args()

    return args


def generate_masks(generator_func, count, output_path):
    name_len = len(str(count))

    for i in range(count):
        name = str(i).zfill(name_len)
        path = output_path.joinpath(f'{name}.png')
        mask = generator_func()
        Image.fromarray(np.uint8(mask)).save(path)


def main():
    args = parse_args()
    image_size = (args.height, args.width)
    percent_range = (args.lower, args.upper)
    noise_percent = args.percent
    generator = MaskGenerator(seed=args.seed)
    output_path = args.output_path
    generator_types = {
        'rr': lambda: generator.random_rectangle(image_size, percent_range),
        'cr': lambda: generator.random_rectangle(image_size, percent_range),
        'noise': lambda: generator.random_noise(image_size, noise_percent)
    }

    output_path.mkdir(parents=True, exist_ok=True)
    generate_masks(generator_types[args.type], args.count, args.output_path)


if __name__ == "__main__":
    main()
