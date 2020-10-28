from torch import nn
from .DCGAN_config import _C


class GNet(nn.Module):
    def __init__(self): # , ngpu):
        super(GNet, self).__init__()
        self.config = _C

        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.config.dimLatentVector, out_channels=self.config.dimG * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimG * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.config.dimG * 8, out_channels=self.config.dimG * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimG * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=self.config.dimG * 4, out_channels=self.config.dimG * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimG * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=self.config.dimG * 2, out_channels=self.config.dimG,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimG),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=self.config.dimG, out_channels=self.config.dimOutput,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, nn_input):
        return self.main(nn_input)


class DNet(nn.Module):
    def __init__(self): # , ngpu):
        super(DNet, self).__init__()
        self.config = _C

        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.config.dimOutput, out_channels=self.config.dimD, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=self.config.dimD, out_channels=self.config.dimD * 2, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimD * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=self.config.dimD * 2, out_channels=self.config.dimD * 4, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimD * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=self.config.dimD * 4, out_channels=self.config.dimD * 8, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=self.config.dimD * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=self.config.dimD * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, nn_input):
        return self.main(nn_input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def getGNet(device):
    # Create the generator
    netG = GNet().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the models
    # print(netG)

    return netG


def getDNet(device):
    netD = DNet().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the models
    # print(netD)

    return netD