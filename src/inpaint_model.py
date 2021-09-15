import torch
from torch import nn
from torch import optim
import numpy as np
from scipy.signal import convolve2d
# from models.DCGAN_trainer import DCGANTrainer
from DCGAN.DCGAN_nets import GNet, DNet
from DCGAN.DCGAN_nets import load_model as load_dcgan
from styleGAN.model import load_model as load_stylegan
from styleGAN.model import get_mean_style
from inpaint_config import _dcgan_inpaint_config as _DIC
from inpaint_config import _stylegan_inpaint_config as _SIC
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from utils.helpers import NormalizeInverse
from utils import helpers
from utils import poissonblending
import math
from tqdm import tqdm

class InpaintModel:
    def __init__(self, model_filename, config, gan_type):
        self.model_config = config
        self.gan_type = gan_type
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.l = 0.003

        self.G, self.D = self.load_model(model_filename)

        # StyleGAN
        if self.gan_type == 'stylegan':
            self.config = _SIC
            self.mean_style = get_mean_style(self.G, self.device)
            self.step = int(math.log(self.model_config.imageSize, 2)) - 2

        if self.gan_type == 'dcgan':
            self.config = _DIC

    def load_model(self, model_filename):
        if self.gan_type == 'stylegan':
            return load_stylegan(model_filename, device=self.device)

        return load_dcgan(model_filename, device=self.device)

    def sample_generator(self, z):
        if self.gan_type == 'stylegan':
            return self.G(
                z,
                step=self.step,
                alpha=self.config.alpha,
                mean_style=self.mean_style,
                style_weight=self.config.style_weight,)

        return self.G(z)

    def sample_discriminator(self, input):
        if self.gan_type == 'stylegan':
            return self.D(input, step=self.step, alpha=1)

        return self.D(input)

    def inpaint_loss(self, W, G_output, y):
        context_loss = torch.sum(
            torch.flatten(
                torch.abs(torch.mul(W, G_output) - torch.mul(W, y))
            )
        )

        dgo = self.sample_discriminator(G_output).view(-1)
        prior_loss = torch.mul(torch.sum(torch.log(1 - dgo)), self.l)

        loss = context_loss + prior_loss

        return loss

    # def create_importance_weights(self, mask, w_size=7):
    #     mask_2d = mask[0, :, :].cpu().numpy()
    #     kernel = np.ones((w_size, w_size), dtype=np.float32)
    #     kernel = kernel / np.sum(kernel)
    #
    #     importance_weights = mask_2d * convolve2d(1 - mask_2d, kernel, mode='same', boundary='symm')
    #
    #     return torch.from_numpy(np.repeat(importance_weights[np.newaxis, :, :], 3, axis=0)).to(self.device)

    def create_importance_weights(self, masks, w_size=7):
        masks = masks.cpu()
        mask_importance_weights = []
        masks_2d = masks[:, 0, :, :]

        for mask in masks_2d:
            kernel = torch.ones((w_size, w_size))
            kernel = kernel / torch.sum(kernel)

            importance_weights = mask * convolve2d(1 - mask, kernel, mode='same', boundary='symm')
            mask_importance_weights.append(importance_weights.unsqueeze(0).repeat(3, 1, 1))

        return torch.stack(mask_importance_weights).to(self.device)

    def preprocess(self, corrupted_images, image_masks):
        # resize_transform = transforms.Compose([
        #     transforms.Resize(self.model_config.imageSize),
        #     transforms.CenterCrop(self.model_config.imageSize),
        #     transforms.ToTensor(),
        # ])
        normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        images = normalize_transform(corrupted_images).to(self.device)
        masks = image_masks.to(self.device)
        masks = helpers.binarize_masks(masks)
        masks = helpers.create_3channel_masks(masks)

        return images, masks

    def postprocess(self, generated_output, corrupted_images, image_masks):
        inverse_normalization = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # Without blending
        # inpainted_image = (image_mask * masked_image) + ((1 - image_mask) * generated_output)
        # inpainted_image = inverse_normalization(inpainted_image.squeeze(dim=0)).permute(1, 2, 0).cpu().numpy()

        generated_images = inverse_normalization(generated_output).permute(0, 2, 3, 1).cpu().numpy()
        corrupted_images = inverse_normalization(corrupted_images).permute(0, 2, 3, 1).cpu().numpy()
        masks = image_masks.permute(0, 2, 3, 1).cpu().numpy()

        inpainted_images = np.empty_like(generated_images)
        for i in range(len(generated_output)):
            inpainted_images[i] = poissonblending.blend(corrupted_images[i], generated_images[i], 1 - masks[i])

        return generated_images, inpainted_images, masks

    def inpaint(self, corrupted_images, image_masks):
        corrupted_images, image_masks = self.preprocess(corrupted_images, image_masks)
        W = self.create_importance_weights(image_masks, w_size=self.config.w_size)

        if self.gan_type == 'stylegan':
            z = nn.Parameter(torch.randn((corrupted_images.size(0), self.model_config.dimLatentVector), device=self.device),
                             requires_grad=True)
        else:
            z = nn.Parameter(torch.randn((corrupted_images.size(0), self.model_config.dimLatentVector, 1, 1), device=self.device),
                             requires_grad=True)

        # z = nn.Parameter(torch.randn((1, self.config.dimLatentVector, 1, 1), device=self.device), requires_grad=True)
        optimizer = optim.Adam([z])

        for i in tqdm(range(self.config.iter)):
            optimizer.zero_grad()
            # G_output = self.G(z)
            G_output = self.sample_generator(z)
            loss = self.inpaint_loss(W, G_output, corrupted_images)
            loss.backward()
            optimizer.step()

        # G_z = self.G(z)
        G_z = self.sample_generator(z)
        G_z_image, inpainted_image, mask = self.postprocess(G_z.detach(), corrupted_images, image_masks)
        importance_weight = W.permute(0, 2, 3, 1).cpu().numpy()

        return importance_weight, G_z_image, inpainted_image
