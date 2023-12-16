from torch import nn
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# Dataset initialization
transform = transforms.ToTensor()

dataset = np.load('C:/Users/Admin/Desktop/MVA/IIN/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')["imgs"]
dataset=[transform(Image.fromarray(img,mode="L")) for img in dataset]
train_set, test_set = torch.utils.data.random_split(dataset, [0.95,0.05])

from torch.utils.data import DataLoader

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_size=6):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_size)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(-1,512*2*2)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_size=6):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 512 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 2, 2)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))  # Sigmoid activation for the last layer if the output is normalized [0, 1]
        return x

# Discriminator

class MLP_Discriminator(nn.Module):
    def __init__(self, latent_dim=6):
        super(MLP_Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.layers(x)
        return x
    
    def permute_dims(z):
        B, d = z.shape
        for j in range(d):
            pi = np.random.permutation(B)
            z[j] = z[j][pi]
        return z
    
    def discrim_loss(self, discrim_probas, new_discrim_probas):
        loss = 0.5*torch.mean(torch.log(discrim_probas)) - 0.5*torch.mean(torch.log(1-new_discrim_probas))
        return loss


# Combine Encoder, Decoder and Discriminator into Factor-VAE
class Factor_VAE(nn.Module):
    def __init__(self, latent_size=6):
        super(Factor_VAE, self).__init__()
        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z

    def fvae_loss(self, x_recons, x, mu, logvar, gamma, discriminator_proba):
        reproduction_loss = nn.functional.binary_cross_entropy(x_recons, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        MLP_loss = torch.log(discriminator_proba/(1-discriminator_proba))

        return reproduction_loss + KLD - gamma * MLP_loss


def train(model, discrim, model_optimizer, discrim_optimizer, epochs, device="cpu", gamma=4):
    model.train()
    discrim.train()
    for epoch in range(epochs):
        t= time.time()
        overall_vae_loss = 0
        overall_discrim_loss = 0
        for x in train_loader:
            x = x.to(device)

            model_optimizer.zero_grad()
            discrim_optimizer.zero_grad()

            # Update of the VAE parameters

            x_recons, mean, logvar, z = model(x) # Used for the FVAE update

            discrim_probas = discrim(z)

            fvae_loss = model.fvae_loss(x_recons, x, mean, logvar, gamma, discrim_probas)
            
            overall_vae_loss += fvae_loss.item()
            
            fvae_loss.backward()

            model_optimizer.step()

            # Update of the discriminator parameters

            _, mean, logvar, z = model(x) # Used for the MLP update

            z = discrim.permute_dims(z)

            new_discrim_probas = discrim(z)

            discrim_loss = discrim.discrim_loss(discrim_probas, new_discrim_probas)

            overall_discrim_loss += discrim_loss.item()

            discrim_optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage VAE Loss: ", overall_vae_loss / len(train_loader.dataset), "\tAverage MLP Loss: ", overall_discrim_loss / len(train_loader.dataset), "\tDuration: ", time.time()-t)
    return overall_vae_loss, overall_discrim_loss

device = "cuda"

model = Factor_VAE().to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
discrim = MLP_Discriminator().to(device)
discrim_optimizer = torch.optim.Adam(discrim.parameters(), lr=1e-3)

# Training
gamma = 40

train(model, discrim, model_optimizer, discrim_optimizer, 20, device=device, gamma=gamma)

torch.save(model.state_dict(), "C:/Users/Admin/Desktop/MVA/IIN/IIN_VAE_Project/factor_vae_model.pt")
torch.save(discrim.state_dict(), "C:/Users/Admin/Desktop/MVA/IIN/IIN_VAE_Project/factor_vae_discrim.pt")