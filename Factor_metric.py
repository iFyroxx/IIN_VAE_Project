import torch
import numpy as np
from beta_VAE import beta_VAE
from factor_VAE import Factor_VAE
from dae import DAE
from TC_VAE import VAE
from tqdm import tqdm

### SAVE EMPIRICAL STDS OF REPRESENTATION OF THE WHOLE DATA

# Dataset initialization
images = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)
labels = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["latents_values"])

from torch.utils.data import DataLoader

batch_size = 64
imgs_set = DataLoader(images, batch_size=batch_size, shuffle=True)

device = "cuda"
z=4

# for model_name in ["factor", "beta", "DAE", "TC"]:
#     if model_name=="beta":
#         for beta in [1,4]:
#             model = beta_VAE(latent_size=z).to(device)
#             epochs = 500
#             model.load_state_dict(torch.load(f"./beta{beta}_vae_{epochs}_z_{z}.pt"))
#             model.eval()
#             encodings = []
#             for x in tqdm(imgs_set):
#                 x = x.to(device)

#                 encoded_x=model.encoder(x).detach().cpu().numpy()[:,:z]
#                 encodings.append(encoded_x)

#             encodings = np.array(encodings).reshape((-1,z))
#             std = np.std(encodings, axis=0)
#             np.savez_compressed(f"empirical_stds_{model_name}_{beta}_VAE.npz", std=std)
#     elif model_name =="factor":
#         model = Factor_VAE(latent_size=z).to(device)
#         model.load_state_dict(torch.load(f"./factor_vae_model_500_z_{z}.pt"))

#         model.eval()
#         encodings = []
#         for x in tqdm(imgs_set):
#             x = x.to(device)

#             encoded_x=model.encoder(x).detach().cpu().numpy()[:,:z]
#             encodings.append(encoded_x)

#         encodings = np.array(encodings).reshape((-1,z))
#         std = np.std(encodings, axis=0)
#         np.savez_compressed(f"empirical_stds_{model_name}_VAE.npz", std=std)
#     elif model_name =="DAE":
#         alpha = torch.Tensor([[1., 1., 0.01, 0.01]]).cuda()
#         model = DAE(4, alpha).to(device)
#         model.load_state_dict(torch.load(f"./DAE_500.pt"))

#         model.eval()
#         encodings = []
#         for x in tqdm(imgs_set):
#             x = x.to(device)

#             c, encoded_x=model.encode(x)
#             encodings.append(c.detach().cpu().numpy()[:,:2*z])

#         encodings = np.array(encodings).reshape((-1,2*z))
#         std = np.std(encodings, axis=0)
#         np.savez_compressed(f"empirical_stds_{model_name}.npz", std=std)
#     elif model_name =="TC":
#         model = VAE(z,6).to(device)
#         model.load_state_dict(torch.load(f"./TC_VAE_500.pt"))

#         model.eval()
#         encodings = []
#         for x in tqdm(imgs_set):
#             x = x.to(device)

#             encoded_x=model.encoder(x).detach().cpu().numpy()[:,:z]
#             encodings.append(encoded_x)

#         encodings = np.array(encodings).reshape((-1,z))
#         std = np.std(encodings, axis=0)
#         np.savez_compressed(f"empirical_stds_{model_name}_VAE.npz", std=std)


### METRIC IMPLEMENTATION
L=100
M=500
z=4

device ="cuda"
model_name="DAE"
if model_name=="beta":
    beta = 4
    s_beta = torch.tensor(np.load(f"empirical_stds_beta_{beta}_VAE.npz")["std"], device=device)
    model = beta_VAE(latent_size=z).to(device)
    epochs = 500
    model.load_state_dict(torch.load(f"./beta{beta}_vae_{epochs}_z_{z}.pt"))
    model.eval()
    accuracies = np.zeros(10)
    for epoch in range(10):
        possible_factors = torch.tensor([1,3,4,5])
        V = torch.zeros((4,z))
        y_true_idx = torch.ones(4).multinomial(M,replacement=True)
        y_true = possible_factors[y_true_idx] # Choose the fixed parameters, labels for the classifiers
        ds = np.zeros(M)
        for i in tqdm(range(M)):
            zs = torch.zeros((L,z))
            for l in range(L):
                v = torch.randint(len(labels),(1,))[0]
                if l==0:
                    fixed_factor_value = labels[v][y_true[i]]
                while labels[v][y_true[i]]!=fixed_factor_value:
                    v = torch.randint(len(labels),(1,))[0]
                x = images[v].to(device)
                latent = model.encoder(x).detach()
                latent = latent[:, :z]
                latent = (latent/s_beta).squeeze(0)
                zs[l] = latent
            variances = torch.var(zs, axis=0)
            d = torch.argmin(variances) # Inputs for the classifiers
            ds[i] = d.cpu().numpy()
            V[y_true_idx[i],d] += 1
        predictions = possible_factors[torch.argmax(V, dim=0)].cpu().numpy()
        ds = ds.astype(int)
        acc = np.mean(predictions[ds]==y_true.cpu().numpy())
        accuracies[epoch] = acc

elif model_name=="factor":
    s_factor = torch.tensor(np.load("empirical_stds_factor_VAE.npz")["std"], device=device)
    model = Factor_VAE(latent_size=z).to(device)
    model.load_state_dict(torch.load(f"./factor_vae_model_500_z_{z}.pt"))
    model.eval()
    accuracies = np.zeros(10)
    for epoch in range(10):
        possible_factors = torch.tensor([1,3,4,5])
        V = torch.zeros((4,z))
        y_true_idx = torch.ones(4).multinomial(M,replacement=True)
        y_true = possible_factors[y_true_idx] # Choose the fixed parameters, labels for the classifiers
        ds = np.zeros(M)
        for i in tqdm(range(M)):
            zs = torch.zeros((L,z))
            for l in range(L):
                v = torch.randint(len(labels),(1,))[0]
                if l==0:
                    fixed_factor_value = labels[v][y_true[i]]
                while labels[v][y_true[i]]!=fixed_factor_value:
                    v = torch.randint(len(labels),(1,))[0]
                x = images[v].to(device)
                latent = model.encoder(x).detach()
                latent = latent[:, :z]
                latent = (latent/s_factor).squeeze(0)
                zs[l] = latent
            variances = torch.var(zs, axis=0)
            d = torch.argmin(variances) # Inputs for the classifiers
            ds[i] = d.cpu().numpy()
            V[y_true_idx[i],d] += 1
        predictions = possible_factors[torch.argmax(V, dim=0)].cpu().numpy()
        ds = ds.astype(int)
        acc = np.mean(predictions[ds]==y_true.cpu().numpy())
        accuracies[epoch] = acc

elif model_name=="DAE":
    s_DAE = torch.tensor(np.load("empirical_stds_DAE.npz")["std"], device=device)
    alpha = torch.Tensor([[1., 1., 0.01, 0.01]]).cuda()
    model = DAE(4, alpha)
    model.load_state_dict(torch.load(f"./DAE_500.pt"))
    model = model.to(device)
    model.eval()
    accuracies = np.zeros(10)
    for epoch in range(10):
        possible_factors = torch.tensor([1,3,4,5])
        V = torch.zeros((4,2*z))
        y_true_idx = torch.ones(4).multinomial(M,replacement=True)
        y_true = possible_factors[y_true_idx] # Choose the fixed parameters, labels for the classifiers
        ds = np.zeros(M)
        for i in tqdm(range(M)):
            zs = torch.zeros((L,2*z))
            for l in range(L):
                v = torch.randint(len(labels),(1,))[0]
                if l==0:
                    fixed_factor_value = labels[v][y_true[i]]
                while labels[v][y_true[i]]!=fixed_factor_value:
                    v = torch.randint(len(labels),(1,))[0]
                x = images[v].to(device)
                latent,_ = model.encode(x)
                latent = latent.detach()[:, :2*z]
                latent = (latent/s_DAE).squeeze(0)
                zs[l] = latent
            variances = torch.var(zs, axis=0)
            d = torch.argmin(variances) # Inputs for the classifiers
            ds[i] = d.cpu().numpy()
            V[y_true_idx[i],d] += 1
        predictions = possible_factors[torch.argmax(V, dim=0)].cpu().numpy()
        ds = ds.astype(int)
        acc = np.mean(predictions[ds]==y_true.cpu().numpy())
        accuracies[epoch] = acc

elif model_name=="TC":
    s_TC = torch.tensor(np.load("empirical_stds_TC_VAE.npz")["std"], device=device)
    model = VAE(z,6).to(device)
    model.load_state_dict(torch.load(f"./TC_VAE_500.pt"))
    model.eval()
    accuracies = np.zeros(10)
    for epoch in range(10):
        possible_factors = torch.tensor([1,3,4,5])
        V = torch.zeros((4,z))
        y_true_idx = torch.ones(4).multinomial(M,replacement=True)
        y_true = possible_factors[y_true_idx] # Choose the fixed parameters, labels for the classifiers
        ds = np.zeros(M)
        for i in tqdm(range(M)):
            zs = torch.zeros((L,z))
            for l in range(L):
                v = torch.randint(len(labels),(1,))[0]
                if l==0:
                    fixed_factor_value = labels[v][y_true[i]]
                while labels[v][y_true[i]]!=fixed_factor_value:
                    v = torch.randint(len(labels),(1,))[0]
                x = images[v].to(device)
                latent = model.encoder(x).detach()
                latent = latent[:, :z]
                latent = (latent/s_TC).squeeze(0)
                zs[l] = latent
            variances = torch.var(zs, axis=0)
            d = torch.argmin(variances) # Inputs for the classifiers
            ds[i] = d.cpu().numpy()
            V[y_true_idx[i],d] += 1
        predictions = possible_factors[torch.argmax(V, dim=0)].cpu().numpy()
        ds = ds.astype(int)
        acc = np.mean(predictions[ds]==y_true.cpu().numpy())
        accuracies[epoch] = acc

print(accuracies)
print(model_name, np.mean(accuracies), np.std(accuracies))