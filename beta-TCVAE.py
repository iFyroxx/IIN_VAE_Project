from torch import nn
import torch
import numpy as np
import time
from tqdm import tqdm

# Dataset initialization
dataset = torch.tensor(np.load('./dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)

torch.manual_seed(10)
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
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(512*2*2, 2*latent_size)
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
            nn.ReLU(True),
            nn.Flatten(),
            self.fc,
        )

    def forward(self, x):
        z = self.layers(x)
        return z

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_size=6):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 512*2*2)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.layers = nn.Sequential(
            self.fc,
            nn.ReLU(True),
            nn.Unflatten(1,(512,2,2)),
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

# Combine Encoder and Decoder into an Autoencoder
class beta_VAE(nn.Module):
    def __init__(self, latent_size=6):
        super(beta_VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent = self.encoder(x)
        mu = latent[:, :self.latent_size]
        logvar = latent[:, self.latent_size:]
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def loss(self, x_recons, x, mu, logvar, alpha, beta, gamma):
        reproduction_loss = nn.functional.binary_cross_entropy(x_recons, x, reduction='sum')
        

        return reproduction_loss

def train(model, optimizer, epochs, device="cpu", beta=4):
    model.train()
    for epoch in range(epochs):
        t= time.time()
        overall_loss = 0
        for x in tqdm(train_set):
            x = x.to(device)

            optimizer.zero_grad()

            x_recons, mean, logvar = model(x)
            loss = model.loss(x_recons, x, mean, logvar, beta)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / len(train_set.dataset), "\tDuration: ", time.time()-t)
    return overall_loss

device = "cuda"

model = beta_VAE().to(device)
# model.load_state_dict(torch.load("./beta4_vae.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
beta = 4

train(model, optimizer, 30, device=device, beta = beta)

torch.save(model.state_dict(), "./beta4_vae.pt")