# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from model import Generatorcifar, Discriminator, weights_init
from data_pipeline import get_cifar10_dataloader
from DiffAugment_pytorch import DiffAugment

# --- CONFIGURATION ---
dataroot = "data"
image_size = 64
batch_size = 128
nz = 100  # Size of the latent z vector
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 35  # You can now set a new, higher target epoch
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.

# --- RESUME TRAINING SETUP ---
# Set this to the epoch you want to resume from.
# Set to 0 to start training from scratch.
resume_epoch = 0

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- SETUP ---
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dataloader = get_cifar10_dataloader(batch_size, image_size)

    # --- MODEL INITIALIZATION ---
    netG = Generatorcifar(nz, ngf).to(device)
    netD = Discriminator(ndf=ndf).to(device)

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # --- LOSS AND OPTIMIZERS ---
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # --- LOAD CHECKPOINTS IF RESUMING ---
    start_epoch = 0
    if resume_epoch > 0:
        print(f"Resuming training from epoch {resume_epoch}...")
        try:
            # Load Generator
            g_checkpoint_path = f"models/netG_epoch_{resume_epoch}.pth"
            netG.load_state_dict(torch.load(g_checkpoint_path, map_location=device))

            # Load Discriminator
            d_checkpoint_path = f"models/netD_epoch_{resume_epoch}.pth"
            netD.load_state_dict(torch.load(d_checkpoint_path, map_location=device))

            start_epoch = resume_epoch
            print("Successfully loaded model checkpoints.")
        except FileNotFoundError:
            print(
                f"Checkpoint files not found for epoch {resume_epoch}. Starting from scratch."
            )
            start_epoch = 0

    # --- TRAINING LOOP ---
    print("Starting Training Loop...")
    policy = "color,translation,cutout"

    # The loop now starts from 'start_epoch' instead of 0
    for epoch in range(start_epoch, num_epochs):
        for i, data in enumerate(dataloader, 0):
            # ... (The entire training logic for Discriminator and Generator remains the same)
            # 1. Update Discriminator network
            netD.zero_grad()
            real_cpu = data[0].to(device)
            real_augmented = DiffAugment(real_cpu, policy=policy)
            b_size = real_augmented.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_augmented).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            fake_augmented = DiffAugment(fake, policy=policy)
            label.fill_(fake_label)

            output = netD(fake_augmented.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # 2. Update Generator network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_augmented).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(
                    f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

        # After each epoch, save generated images and model checkpoints
        # The epoch number is now correctly handled
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        vutils.save_image(
            fake_samples, f"output/fake_samples_epoch_{epoch+1:03d}.png", normalize=True
        )
        torch.save(netG.state_dict(), f"models/netG_epoch_{epoch+1}.pth")
        torch.save(netD.state_dict(), f"models/netD_epoch_{epoch+1}.pth")

    print("Training finished.")
