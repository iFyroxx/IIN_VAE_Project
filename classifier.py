import torch
import torch.nn as nn
import numpy as np
from beta_VAE import beta_VAE
from tqdm import tqdm

# Dataset initialization
images = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)
labels = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["latents_values"])

class LinearClassifier(nn.Module):
    def __init__(self, in_features = 4, out_features = 4):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return nn.functional.softmax(self.linear(x),dim=1)

B=512
L=100
device = "cuda"
torch.manual_seed(10)

if __name__=="__main__":
    z=10
    for beta in [1,4,10]:
        classifier = LinearClassifier(in_features=z).to(device)

        # Define a loss function and optimizer
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        model = beta_VAE(latent_size=z).to(device)
        model.load_state_dict(torch.load(f"./beta{beta}_vae_500_z_{z}.pt"))
        # Train the classifier on the training data
        classifier.train()
        model.eval()
        for epoch in range(5):
            z_diff = torch.zeros((B, z)).to(device)
            possible_factors = torch.tensor([1,3,4,5])
            y_true_idx = torch.ones(4).multinomial(B,replacement=True)
            y_true = possible_factors[y_true_idx]
            for b in range(B):
                z_diff_b = torch.zeros(1,z).to(device)
                for l in range(L):
                    v1 = torch.randint(len(labels),(1,))[0]
                    v2 = torch.randint(len(labels),(1,))[0]
                    while v2==v1 or labels[v2][y_true[b]]!=labels[v1][y_true[b]]:
                        v2 = torch.randint(len(labels),(1,))[0]
                    x1 = images[v1].to(device)
                    x2 = images[v2].to(device)
                    latent1 = model.encoder(x1).detach()
                    latent2 = model.encoder(x2).detach()
                    z1 = latent1[:, :z]
                    z2 =latent2[:, :z]
                    z_diff_b += torch.abs(z1-z2)
                z_diff_b = z_diff_b/L
                z_diff[b] = z_diff_b
            
            # Forward pass
            y_probs = classifier(z_diff)
            y_pred_idx = torch.argmax(y_probs,dim=1).to(device="cpu")
            y_pred = possible_factors[y_pred_idx]

            # Compute the loss
            loss = loss_fn(y_probs, y_true_idx.to(device=device,dtype=torch.long))
            acc = torch.eq(y_true, y_pred).sum().item()/ y_true.shape[0] * 100

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch%1==0:

                print("Epoch :", epoch, "\tLoss :", loss.item(), "\tAccuracy :", acc)


        torch.save(classifier.state_dict(), f"./classifier_{beta}_z_{z}.pt")

