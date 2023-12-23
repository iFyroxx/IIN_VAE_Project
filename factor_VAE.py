import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch import nn
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# Dataset initialization
dataset = torch.tensor(np.load('C:/Users/Admin/Desktop/MVA/IIN/IIN_VAE_Project/dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)

train_set, test_set = torch.utils.data.random_split(dataset, [0.95,0.05])

from torch.utils.data import DataLoader

batch_size = 64
train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_size=6):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128*2*2, 2*latent_size)
        self.layers = nn.Sequential(
            self.conv1,
            nn.ReLU(True),
            self.conv2,
            nn.ReLU(True),
            self.conv3,
            nn.ReLU(True),
            self.conv4,
            nn.ReLU(True),
            self.conv5,
            nn.ReLU(True)
        )

    def forward(self, x):
        z = self.layers(x)
        z = z.view(-1, 128*2*2)
        z = self.fc(z)
        return z

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_size=6):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 128*2*2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.layers = nn.Sequential(
            self.fc,
            nn.ReLU(True),
            nn.Unflatten(1,(128,2,2)),
            self.deconv1,
            nn.ReLU(True),
            self.deconv2,
            nn.ReLU(True),
            self.deconv3,
            nn.ReLU(True),
            self.deconv4,
            nn.ReLU(True),
            self.deconv5,
            nn.Sigmoid()
        )

    def forward(self, z):
        x_recons = self.layers(z)
        return x_recons

# Discriminator

class MLP_Discriminator(nn.Module):
    def __init__(self, latent_dim=6):
        super(MLP_Discriminator, self).__init__()
        self.latent_size = latent_dim
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64,128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128,128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1,self.latent_size)
        x = self.layers(x)
        return x
    
    def permute_dims(self,z):
        assert z.dim() == 2

        B, d = z.size()
        perm_z = torch.zeros_like(z)
        for j in range(d):
            perm = torch.randperm(B)
            perm_z[:,j] = z[perm,j]

        return perm_z
    
    def discrim_loss(self, discrim_probas, new_discrim_probas):
        zeros = torch.zeros((batch_size,1), dtype=torch.float, device=device)
        loss = -0.5*torch.mean(torch.log(discrim_probas) + torch.log(1-new_discrim_probas))
        return loss

# Combine Encoder, Decoder and Discriminator into Factor-VAE
class Factor_VAE(nn.Module):
    def __init__(self, latent_size=6):
        super(Factor_VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, no_dec = False):
        latent = self.encoder(x)
        mu = latent[:, :self.latent_size]
        logvar = latent[:, self.latent_size:]
        z = self.reparameterize(mu, logvar)
        if no_dec:
            return z.detach()
        else:
            reconstructed = self.decoder(z)
            return reconstructed, mu, logvar, z
    
    def fvae_loss(self, x_recons, x, mu, logvar, gamma, discriminator_probas):
        reproduction_loss = nn.functional.binary_cross_entropy(x_recons, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        MLP_loss = torch.mean(torch.log(discriminator_probas) - torch.log(1-discriminator_probas))

        return reproduction_loss + KLD - gamma * MLP_loss

from tqdm import tqdm

def train(model, discrim, model_optimizer, discrim_optimizer, epochs, device="cpu", gamma=4):
    model.train()
    discrim.train()

    for epoch in range(epochs):
        t = time.time()
        overall_vae_loss = 0
        overall_discrim_loss = 0
        for i in range(0, len(train_set), 2):
            x1 = next(iter(train_set))
            x2 = next(iter(train_set))
            x1 = x1.to(device)
            x2 = x2.to(device)

            # Update of the VAE parameters
            x_recons, mean, logvar, z = model(x1)  # Used for both the FVAE update and MLP update

            discrim_probas = discrim(z).detach()

            fvae_loss = model.fvae_loss(x_recons, x1, mean, logvar, gamma, discrim_probas)
            
            overall_vae_loss = overall_vae_loss + fvae_loss.item()

            model_optimizer.zero_grad()
            fvae_loss.backward(retain_graph=True)
            model_optimizer.step()

            # Update of the discriminator parameters
            z_prime = model(x2, no_dec= True)  # Used for the MLP update

            z2 = discrim.permute_dims(z_prime).detach()

            new_discrim_probas = discrim(z2)

            discrim_loss = discrim.discrim_loss(discrim_probas, new_discrim_probas)

            overall_discrim_loss = overall_discrim_loss + discrim_loss.item()

            discrim_optimizer.zero_grad()
            discrim_loss.backward()
            discrim_optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage VAE Loss: ", overall_vae_loss / len(train_set.dataset), "\tAverage MLP Loss: ", overall_discrim_loss / len(train_set.dataset), "\tDuration: ", time.time() - t)
    return overall_vae_loss, overall_discrim_loss

device = "cuda"

if __name__=="__main__":
    z = 10
    model = Factor_VAE(latent_size=z).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    discrim = MLP_Discriminator(latent_dim=z).to(device)
    discrim_optimizer = torch.optim.Adam(discrim.parameters(), lr=1e-5)

    # Training
    gamma = 40

    print("Training starting")

    train(model, discrim, model_optimizer, discrim_optimizer, epochs=500, device=device, gamma=gamma)

    torch.save(model.state_dict(), f"./factor_vae_model_z_{z}.pt")
    torch.save(discrim.state_dict(), f"./factor_vae_discrim_z_{z}.pt")