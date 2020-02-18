import torch
from torch import nn
from torch import optim
import numpy as np
from scipy.signal import convolve2d
from models.DCGAN_trainer import DCGANTrainer
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
from PIL import Image


class InpaintModel:
    def __init__(self, model):
        self.config = model.get_config()
        self.model = model
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.G = self.model.getNetG()
        self.D = self.model.getNetD()
        self.l = 0.003

    def inpaint_loss(self, W, G_output, y):
        W = W.to(self.device)
        G_output = G_output.to(self.device)
        y = y.to(self.device)

        context_loss = torch.sum(
            torch.flatten(
                torch.abs(torch.mul(W, G_output) - torch.mul(W, y))
            )
        )

        dgo = self.D(G_output).view(-1)
        prior_loss = torch.mul(torch.log(1 - dgo), self.l)

        loss = context_loss + prior_loss

        return loss

    def create_importance_weights(self, mask, w_size=7):
        mask_2d = mask[0, :, :]
        kernel = np.ones((w_size, w_size), dtype=np.float32)
        kernel = kernel / np.sum(kernel)

        importance_weights = convolve2d(mask_2d, kernel, mode='same')  # , boundary='symm')
        importance_weights[mask_2d == 1] = 0

        return torch.from_numpy(np.repeat(importance_weights[np.newaxis, :, :], 3, axis=0))

    def preprocess(self, masked_image, image_mask):
        resize_transform = transforms.Compose([
            transforms.Resize(self.config.imageSize),
            transforms.CenterCrop(self.config.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = resize_transform(masked_image)  # .to(self.device)
        mask = resize_transform(image_mask)  # .to(self.device)

        return image, mask

    def inpaint(self, masked_image, image_mask):
        image, mask = self.preprocess(masked_image, image_mask)
        W = self.create_importance_weights(mask.numpy())

        z = nn.Parameter(torch.randn(1, self.config.dimLatentVector, 1, 1, device=self.device), requires_grad=True)
        optimizer = optim.Adam([z])

        for i in range(1500):
            optimizer.zero_grad()
            G_output = self.G(z)
            loss = self.inpaint_loss(W, G_output, image)
            loss.backward()
            optimizer.step()

        return self.G(z)
