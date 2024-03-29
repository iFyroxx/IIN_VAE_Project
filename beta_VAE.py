from torch import nn
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output

# Dataset initialization
dataset_imgs = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)

torch.manual_seed(10)
imgs_train_set, imgs_val_set, imgs_test_set = torch.utils.data.random_split(dataset_imgs, [0.8,0.1,0.1])

from torch.utils.data import DataLoader

batch_size = 64
imgs_train_set = DataLoader(imgs_train_set, batch_size=batch_size, shuffle=True)
imgs_val_set = DataLoader(imgs_val_set, batch_size=batch_size, shuffle=True)
imgs_test_set = DataLoader(imgs_test_set, batch_size=batch_size, shuffle=True)

### Model definition ###

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
    
    def loss(self, x_recons, x, mu, logvar, beta):
        reproduction_loss = nn.functional.binary_cross_entropy(x_recons, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())

        return reproduction_loss + beta*KLD

### Train function ###

def train(model, optimizer, epochs, device="cpu", beta=4):
    train_losses = []
    val_losses = []
    fig = plt.figure()
    for epoch in range(epochs):
        t= time.time()
        overall_loss = 0
        model.train()
        for x in imgs_train_set:
            x = x.to(device)

            optimizer.zero_grad()

            x_recons, mean, logvar = model(x)
            loss = model.loss(x_recons, x, mean, logvar, beta)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        train_losses.append(overall_loss/len(imgs_train_set.dataset))
        model.eval()
        overall_eval_loss = 0
        for x in imgs_val_set:
            x = x.to(device)
            x_recons, mean, logvar = model(x)
            loss = model.loss(x_recons, x, mean, logvar, beta)
            overall_eval_loss += loss.item()
        val_losses.append(overall_eval_loss/len(imgs_val_set.dataset))

        if epoch%10==0 and epoch>51: 
            plt.clf()
            plt.plot(np.arange(epoch-50,epoch), train_losses[-50:], color="blue")
            plt.plot(np.arange(epoch-50,epoch), val_losses[-50:], color="orange")
            fig.canvas.draw()
            plt.pause(0.05)

        
        if epoch%10==0:
            print("\tEpoch", epoch + 1, "\tAverage train Loss: ", overall_loss / len(imgs_train_set.dataset),"\tAverage val Loss: ", overall_eval_loss / len(imgs_val_set.dataset), "\tDuration: ", time.time()-t)
            torch.save(model.state_dict(), f"./beta{beta}_vae_{epoch}_z_{z}.pt")
            if os.path.exists(f"./beta{beta}_vae_{epoch-10}_z_{z}.pt"):
                os.remove(f"./beta{beta}_vae_{epoch-10}_z_{z}.pt")
    return overall_loss

### Training loop ###

device = "cuda"

if __name__=="__main__":
    for beta in [1,4]:
        z = 4
        model = beta_VAE(latent_size=z).to(device)
        # model.load_state_dict(torch.load("./beta4_vae.pt"))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        epochs = 500
        train(model, optimizer, epochs, device=device, beta = beta)

        model.eval()
        loss = nn.BCELoss()
        overall_loss = 0
        for x in imgs_test_set:
            x=x.to(device)
            x_recon,_,_ = model(x)
            overall_loss += loss(x_recon, x).item()
        
        print(overall_loss/len(imgs_test_set.dataset)*64)

        os.remove(f"./beta{beta}_vae_{epochs-10}_z_{z}.pt")
        torch.save(model.state_dict(), f"./beta{beta}_vae_{epochs}_z_{z}.pt")