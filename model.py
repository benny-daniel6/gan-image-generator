# model.py

import torch.nn as nn


def weights_init(m):
    """
    Custom weights initialization called on netG and netD to help models converge.
    From the DCGAN paper, the authors specify that all model weights shall be
    randomly initialized from a Normal distribution with mean=0, stdev=0.02.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# use this for original DCGAN Generator for cifar10
class Generatorcifar(nn.Module):
    """
    The Generator network takes a latent vector (z) as input and generates an image.
    It uses a series of transposed convolutions to upsample the vector to a 64x64x3 image.
    """

    def __init__(self, nz=100, ngf=64, nc=3):
        """
        Args:
            nz (int): Size of the latent z vector (input noise).
            ngf (int): Size of feature maps in the generator.
            nc (int): Number of channels in the output image (3 for RGB).
        """
        super(Generatorcifar, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Final state size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    # --------------use this for modified DCGAN Generator for celeba dataset----------------


class Generatorceleb(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generatorceleb, self).__init__()
        # NOTE: This should be 'self.main' or 'self.net' depending on your last fix
        self.net = nn.Sequential(
            # I have removed 'bias=False' from all layers below
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.net(input)


class Discriminator(nn.Module):
    """
    The Discriminator network takes an image as input and classifies it as real or fake.
    It's a standard CNN that outputs a single probability.
    """

    def __init__(self, nc=3, ndf=64):
        """
        Args:
            nc (int): Number of channels in the input image (3 for RGB).
            ndf (int): Size of feature maps in the discriminator.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
