from torch import nn
from .DCGAN_config import _C


class GNet(nn.Module):
    def __init__(self): # , ngpu):
        super(GNet, self).__init__()
        self.config = _C

        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.dimLatentVector, self.config.dimG * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.config.dimG * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.config.dimG * 8, self.config.dimG * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.dimG * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.config.dimG * 4, self.config.dimG * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.dimG * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.config.dimG * 2, self.config.dimG, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.dimG),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.config.dimG, self.config.dimOutput, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DNet(nn.Module):
    def __init__(self): # , ngpu):
        super(DNet, self).__init__()
        self.config = _C

        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.config.dimOutput, self.config.dimD, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.config.dimD, self.config.dimD * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.dimD * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.config.dimD * 2, self.config.dimD * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.dimD * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.config.dimD * 4, self.config.dimD * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.config.dimD * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.config.dimD * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

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
    print(netG)

def getDNet(device):
    netD = DNet().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the models
    print(netD)