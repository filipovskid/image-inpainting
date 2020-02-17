import torch
from torch import nn
from torch import optim
from .DCGAN_config import _C
import torchvision.utils as vutils
from .DCGAN_nets import getGNet, getDNet
from datasets.celeba_dataset import get_image_dataset
from utils.config import printConfig

from pathlib import Path


class DCGANTrainer:
    def __init__(self, data_root, checkpoints_path=None, save_epoch=10, num_epochs=None):
        self.config = _C
        self.checkpoints_path = Path(checkpoints_path)
        self.save_epoch = save_epoch
        if num_epochs:
            self.config.nEpoch = num_epochs
        self.num_epochs = self.config.nEpoch
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.netG = getGNet(self.device)
        self.netD = getDNet(self.device)
        self.dataset = get_image_dataset(data_root)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=_C.miniBatchSize,
                                                      shuffle=True, num_workers=2)

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.baseLearningRate,
                                     betas=(self.config.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.baseLearningRate,
                                     betas=(self.config.beta1, 0.999))

        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.start_epoch = 1

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.config.dimLatentVector, 1, 1, device=self.device)

        printConfig(self.config)

    def train(self):
        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Lists to keep track of progress
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.config.dimLatentVector, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.num_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            if self.checkpoints_path and ((epoch % self.save_epoch == 0) or (epoch == self.num_epochs)):
                self.save_checkpoint(epoch)

        return self.G_losses, self.D_losses, self.img_list

    def save_checkpoint(self, epoch):
        checkpoint = {
            'G': {
                'model_state_dict': self.netG.state_dict(),
                'optim_state_dict': self.optimizerG.state_dict()
            },
            'D': {
                'model_state_dict': self.netD.state_dict(),
                'optim_state_dict': self.optimizerD.state_dict()
            },
            'epoch': epoch,
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
            'img_list': self.img_list,
            'fixed_noise': self.fixed_noise
        }

        checkpoint_name = f'checkpoint_{epoch}.tar'
        checkpoint_path = self.checkpoints_path.joinpath(checkpoint_name)

        torch.save(checkpoint, str(checkpoint_path))

        print(f'[+] Checkpoint saved! Epoch {epoch}.')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.netG.load_state_dict(checkpoint['G']['model_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['G']['optim_state_dict'])

        self.netD.load_state_dict(checkpoint['D']['model_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['D']['optim_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.G_losses = checkpoint['G_losses']
        self.D_losses = checkpoint['D_losses']
        self.img_list = checkpoint['img_list']
        self.fixed_noise = checkpoint['fixed_noise']

        print(f'[+] Checkpoint loaded! Epoch {checkpoint["epoch"]}.')
