import torch
from torch import nn
from torch import optim
import numpy as np
from scipy.signal import convolve2d
from models.DCGAN import DCGAN
import torch.utils.model_zoo as model_zoo
from PIL import Image


class InpaintModel:
    def __init__(self):
        # use_gpu = True if torch.cuda.is_available() else False
        # self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)
        self.model = DCGAN(useGPU=False,
                      storeAVG=True) #,
                      #**kwargs['config'])

        checkpoint = 'https://dl.fbaipublicfiles.com/gan_zoo/DCGAN_fashionGen-1d67302.pth'
        state_dict = model_zoo.load_url(checkpoint, map_location='cpu')
        self.model.load_state_dict(state_dict)

        self.G = self.model.getNetG()
        self.D = self.model.getNetD()
        self.l = 0.003

    def inpaint_loss(self, W, G_output, y):
        context_loss = torch.sum(
            torch.flatten(
                torch.abs(torch.mul(W, G_output) - torch.mul(W, y))
            )
        )

        dgo = self.D(G_output)
        prior_loss = torch.mul(torch.log(1 - dgo), self.l)

        loss = context_loss + prior_loss

        return loss

    def create_importance_weights(self, mask, w_size):
        mask_2d = mask[:, :, 0]
        kernel = np.ones((w_size, w_size), dtype=np.float32)
        kernel = kernel / np.sum(kernel)

        importance_weights = mask_2d * convolve2d(mask_2d, kernel, mode='same')  # , boundary='symm')

        return importance_weights

    def inpaint(self, image, mask):
        W = torch.from_numpy(self.create_importance_weights(mask, 7))
        z = nn.Parameter(torch.FloatTensor(np.random.randn(64, 120)))
        optimizer = optim.Adam([z])

        for i in range(1500):
            optimizer.zero_grad()
            G_output = torch.from_numpy(self.G(z.numpy()))
            loss = self.inpaint_loss(W, G_output, image)
            loss.backward()
            optimizer.step()

        return self.G(z)
