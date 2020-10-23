import torch
from torch import nn
from torch import optim
import numpy as np
from scipy.signal import convolve2d
# from models.DCGAN_trainer import DCGANTrainer
from models.DCGAN_nets import GNet, DNet
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


class InpaintModel:
    def __init__(self, model_filename, config):
        self.config = config
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.G = GNet()
        self.D = DNet()
        self.load_model(model_filename)

        self.l = 0.003

    def load_model(self, model_filename):
        checkpoint = torch.load(model_filename)

        self.G.load_state_dict(checkpoint['G']['model_state_dict'])
        self.D.load_state_dict(checkpoint['D']['model_state_dict'])

        print(f'[+] Model loaded! Epoch {checkpoint["epoch"]}.')

    def inpaint_loss(self, W, G_output, y):
        context_loss = torch.sum(
            torch.flatten(
                torch.abs(torch.mul(W, G_output) - torch.mul(W, y))
            )
        )

        dgo = self.D(G_output).view(-1)
        prior_loss = torch.mul(torch.log(1 - dgo), self.l)

        loss = context_loss + prior_loss

        return loss

    def create_importance_weights(self, mask, w_size=20):
        mask_2d = mask[0, :, :]
        kernel = np.ones((w_size, w_size), dtype=np.float32)
        kernel = kernel / np.sum(kernel)

        importance_weights = mask_2d * convolve2d(1 - mask_2d, kernel, mode='same', boundary='symm')

        return torch.from_numpy(np.repeat(importance_weights[np.newaxis, :, :], 3, axis=0)).to(self.device)

    def preprocess(self, masked_image, image_mask):
        resize_transform = transforms.Compose([
            transforms.Resize(self.config.imageSize),
            transforms.CenterCrop(self.config.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image = normalize_transform(resize_transform(masked_image)).to(self.device)
        mask = resize_transform(image_mask).to(self.device)

        return image, mask

    def postprocess(self, generated_output, masked_image, image_mask):
        generated_img = generated_output.permute(1, 2, 0).numpy()
        mask = image_mask.permute(1, 2, 0).numpy()

        inpainted_image = (mask * masked_image) + ((1 - mask) * generated_img)

        return inpainted_image, generated_img, mask

    def inpaint(self, masked_image, image_mask):
        image, mask = self.preprocess(masked_image, image_mask)
        W = self.create_importance_weights(mask)

        z = nn.Parameter(torch.randn((1, self.config.dimLatentVector, 1, 1), device=self.device), requires_grad=True)
        optimizer = optim.Adam([z])

        for i in range(1500):
            optimizer.zero_grad()
            G_output = self.G(z)
            loss = self.inpaint_loss(W, G_output, image)
            loss.backward()
            optimizer.step()

        G_z = self.G(z)

        return self.postprocess(generated_output=G_z, masked_image = masked_image, image_mask=mask), \
               W.permute(1, 2, 0).numpy()
