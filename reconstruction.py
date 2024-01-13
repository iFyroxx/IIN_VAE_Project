from classifier_metric import LinearClassifier
import torch
import numpy as np
from beta_VAE import beta_VAE
from factor_VAE import Factor_VAE
from dae import DAE
from TC_VAE import VAE
import matplotlib.pyplot as plt

images = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float)
labels = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["latents_values"])

device = "cpu"

model_type = "TC"

if model_type=="beta":
    # For beta-VAE
    beta=4
    z = 4
    model = beta_VAE(latent_size=z).to(device)
    model.load_state_dict(torch.load(f"./beta{beta}_vae_500_z_{z}.pt",map_location=torch.device("cpu")))

elif model_type=="factor":
    # For Factor-VAE
    z = 4
    model = Factor_VAE(latent_size=z).to(device)
    model.load_state_dict(torch.load(f"./factor_vae_model_500_z_{z}.pt",map_location=torch.device("cpu")))

elif model_type=="DAE":
    # For DAE
    z = 4
    alpha = torch.Tensor([[1., 1., 0.01, 0.01]]).to(device)
    model = DAE(z, alpha).to(device)
    model.load_state_dict(torch.load(f"./DAE_500.pt"))

elif model_type=="TC":
    # For TC-VAE
    z = 4
    model = VAE(z,6).to(device)
    model.load_state_dict(torch.load(f"./TC_VAE_500.pt"))

x1 = images[20000].unsqueeze(0).to(device)
x2 = images[60000].unsqueeze(0).to(device)
x3 = images[85000].unsqueeze(0).to(device)
model.eval()
if model_type=="TC":
    x_recons1 = model.reconstruct_img(x1)[0].detach()
    x_recons2 = model.reconstruct_img(x2)[0].detach()
    x_recons3 = model.reconstruct_img(x3)[0].detach()
else:
    x_recons1 = model(x1)[0].detach()
    x_recons2 = model(x2)[0].detach()
    x_recons3 = model(x3)[0].detach()

plt.figure(1)
plt.subplot(1, 3, 1)
plt.imshow(x_recons1.squeeze(0,1),cmap="gray")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(x_recons2.squeeze(0,1),cmap="gray")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(x_recons3.squeeze(0,1),cmap="gray")
plt.axis("off")
plt.show()