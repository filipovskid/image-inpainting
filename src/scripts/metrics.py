import argparse
import pathlib
import imageio
import numpy as np
import pandas as pd

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray


def parse_args():
    parser = argparse.ArgumentParser(description='Script to compute all statistics')
    parser.add_argument('--gt-path', help='Path to ground truth data', type=pathlib.Path, dest='gt_path')
    parser.add_argument('--predicted-path', help='Path to inpainted data', type=pathlib.Path, dest='predicted_path')
    parser.add_argument('--win-size', help='SSIM window size', type=int, default=11, dest='win_size')
    parser.add_argument('--output-path', help='Path where to output data (csv)', type=pathlib.Path, dest='output_path')
    args = parser.parse_args()

    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)

    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))


gt_path = args.gt_path
predicted_path = args.predicted_path
output_path = args.output_path
output_name = output_path.name
output_name = output_name if output_name.split('.')[-1] == 'csv' else output_name + '.csv'

names = []
mae = []
psnr = []
ssim = []


gt_files = sorted([x for x in gt_path.glob('**/*') if x.is_file()])
for gt_file in gt_files:
    name = gt_file.name
    names.append(name)

    image_gt = (imageio.imread(gt_file) / 255.0).astype(np.float32)
    image_pred = (imageio.imread(predicted_path.joinpath(name)) / 255.0).astype(np.float32)

    image_gt = rgb2gray(image_gt)
    image_pred = rgb2gray(image_pred)
    
    mae.append(compare_mae(image_gt, image_pred))
    psnr.append(peak_signal_noise_ratio(image_gt, image_pred, data_range=1))
    ssim.append(structural_similarity(image_gt, image_pred, data_range=1, win_size=args.win_size, multichannel=True))


data = {'name': names, 'MAE': mae, 'PSNR': psnr, 'SSIM': ssim}
df = pd.DataFrame(data)
df.to_csv(output_path.parent.joinpath(output_name), index=False)
