from classifier import LinearClassifier
import torch
import numpy as np
from beta_VAE import beta_VAE
import matplotlib.pyplot as plt

images = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float)
labels = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["latents_values"])

device = "cpu"

beta=4
model = beta_VAE(latent_size=4).to(device)
model.load_state_dict(torch.load(f"./beta{beta}_vae_500.pt",map_location=torch.device("cpu")))

x = images[30000].unsqueeze(0).to(device)
model.eval()
x_recons = model(x)[0].detach()

plt.figure(1)
plt.imshow(x.squeeze(0),cmap="gray")
plt.figure(2)
plt.imshow(x_recons.squeeze(0,1),cmap="gray")
plt.show()