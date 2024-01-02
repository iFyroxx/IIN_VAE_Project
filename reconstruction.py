from classifier_metric import LinearClassifier
import torch
import numpy as np
from beta_VAE import beta_VAE
from factor_VAE import Factor_VAE
import matplotlib.pyplot as plt

images = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float)
labels = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["latents_values"])

print(labels[0],labels[70000])
plt.figure(1)
plt.imshow(images[0].squeeze(0),cmap="gray")
plt.figure(2)
plt.imshow(images[70000].squeeze(0),cmap="gray")
plt.show()

device = "cpu"

model_type = "beta"

if model_type=="beta":
    # For beta-VAE
    beta=10
    z = 10
    model = beta_VAE(latent_size=z).to(device)
    model.load_state_dict(torch.load(f"./beta{beta}_vae_500_z_{z}.pt",map_location=torch.device("cpu")))

elif model_type=="factor":
    # For Factor-VAE
    z = 10
    model = Factor_VAE(latent_size=z).to(device)
    model.load_state_dict(torch.load(f"./factor_vae_model_z_{z}.pt",map_location=torch.device("cpu")))

x = images[90000].unsqueeze(0).to(device)
model.eval()
x_recons = model(x)[0].detach()

plt.figure(1)
plt.imshow(x.squeeze(0),cmap="gray")
plt.figure(2)
plt.imshow(x_recons.squeeze(0,1),cmap="gray")
plt.show()