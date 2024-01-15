import numpy as np


### Keep images that have scale = 0.5

dataset = np.load('C:/Users/Admin/Desktop/MVA/IIN/IIN_VAE_Project/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')
indices = dataset["latents_values"][:,2]==0.5

images = dataset["imgs"][indices]
latent_values = dataset["latents_values"][indices]
metadata = dataset["metadata"]
latent_classes = dataset["latents_classes"][indices]

np.savez_compressed("dsprites_no_scale.npz", metadata=metadata, imgs=images, latents_classes = latent_classes, latents_values=latent_values)