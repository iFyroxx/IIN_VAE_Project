import torch
import torch.nn as nn
import numpy as np
from beta_VAE import beta_VAE
from factor_VAE import Factor_VAE
from tqdm import tqdm

# Dataset initialization
images = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)
labels = torch.tensor(np.load('./dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["latents_values"])

class LinearClassifier(nn.Module):
    def __init__(self, in_features = 4, out_features = 4):
        super(LinearClassifier, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, x):
        return self.linear(x)


device = "cuda"

if __name__=="__main__":
    mode = "eval"
    model = "beta"
    z=10
    if model=="beta":
        if mode =="train":
            B=10
            L=200
            for beta in [1,4]:
                torch.manual_seed(10)
                classifier = LinearClassifier(in_features=z).to(device)

                # Define a loss function and optimizer
                optimizer = torch.optim.Adagrad(classifier.parameters(), lr=1e-2)
                loss_fn = nn.CrossEntropyLoss()
                model = beta_VAE(latent_size=z).to(device)
                model.load_state_dict(torch.load(f"./beta{beta}_vae_500_z_{z}.pt"))
                # Train the classifier on the training data
                classifier.train()
                model.eval()
                total_acc = 0
                for epoch in range(5000):
                    z_diff = torch.zeros((B, z)).to(device)
                    possible_factors = torch.tensor([1,3,4,5])
                    y_true_idx = torch.ones(4).multinomial(B,replacement=True)
                    y_true = possible_factors[y_true_idx] # Choose the fixed parameters
                    for b in range(B):
                        z_diff_b = torch.zeros(1,z).to(device)
                        for l in range(L):
                            v1 = torch.randint(len(labels),(1,))[0]
                            if l==0:
                                fixed_factor_value = labels[v1][y_true[b]]
                            while labels[v1][y_true[b]]!=fixed_factor_value:
                                v1 = torch.randint(len(labels),(1,))[0]
                            v2 = torch.randint(len(labels),(1,))[0]
                            while labels[v2][y_true[b]]!=fixed_factor_value:
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
                    y_logits = classifier(z_diff)
                    y_probs = torch.softmax(y_logits, dim=1)
                    y_pred_idx = torch.argmax(y_probs,dim=1).to(device="cpu")
                    y_pred = possible_factors[y_pred_idx]

                    # Compute the loss
                    loss = nn.functional.cross_entropy(y_logits, y_true_idx.to(device=device,dtype=torch.long))
                    acc = torch.eq(y_true, y_pred).sum().item()/ y_true.shape[0] * 100
                    total_acc+=acc

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch%1==0:

                        print("Epoch :", epoch+1, "\tLoss :", loss.item(), "\tAccuracy :", acc, "\tAverage accuracy :", total_acc/(epoch+1))
                
                torch.save(classifier.state_dict(), f"./classifier_{beta}_z_{z}.pt")
        elif mode =="eval":
            B=800
            L=200
            for beta in [1,4,10]:
                classifier = LinearClassifier(in_features=z).to(device)
                classifier.load_state_dict(torch.load(f"./classifier_{beta}_z_{z}.pt"))

                # Define a loss function and optimizer
                model = beta_VAE(latent_size=z).to(device)
                model.load_state_dict(torch.load(f"./beta{beta}_vae_500_z_{z}.pt"))
                # Train the classifier on the training data
                classifier.eval()
                model.eval()
                z_diff = torch.zeros((B, z)).to(device)
                possible_factors = torch.tensor([1,3,4,5])
                y_true_idx = torch.ones(4).multinomial(B,replacement=True)
                y_true = possible_factors[y_true_idx] # Choose the fixed parameters
                for b in tqdm(range(B)):
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
                y_logits = classifier(z_diff)
                y_probs = torch.softmax(y_logits, dim=1)
                y_pred_idx = torch.argmax(y_probs,dim=1).to(device="cpu")
                y_pred = possible_factors[y_pred_idx]

                # Compute the loss
                acc = torch.eq(y_true, y_pred).sum().item()/ y_true.shape[0] * 100
                print("Beta:",beta, "\tAccuracy :", acc)
    elif model=="factor":
        if mode =="train":
            B=10
            L=200
            torch.manual_seed(10)
            classifier = LinearClassifier(in_features=z).to(device)

            # Define a loss function and optimizer
            optimizer = torch.optim.Adagrad(classifier.parameters(), lr=1e-2)
            loss_fn = nn.CrossEntropyLoss()
            model = Factor_VAE(latent_size=z).to(device)
            model.load_state_dict(torch.load(f"./factor_vae_model_z_{z}.pt"))
            # Train the classifier on the training data
            classifier.train()
            model.eval()
            total_acc = 0
            for epoch in range(5000):
                z_diff = torch.zeros((B, z)).to(device)
                possible_factors = torch.tensor([1,3,4,5])
                y_true_idx = torch.ones(4).multinomial(B,replacement=True)
                y_true = possible_factors[y_true_idx] # Choose the fixed parameters
                for b in range(B):
                    z_diff_b = torch.zeros(1,z).to(device)
                    for l in range(L):
                        v1 = torch.randint(len(labels),(1,))[0]
                        if l==0:
                            fixed_factor_value = labels[v1][y_true[b]]
                        while labels[v1][y_true[b]]!=fixed_factor_value:
                            v1 = torch.randint(len(labels),(1,))[0]
                        v2 = torch.randint(len(labels),(1,))[0]
                        while labels[v2][y_true[b]]!=fixed_factor_value:
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
                y_logits = classifier(z_diff)
                y_probs = torch.softmax(y_logits, dim=1)
                y_pred_idx = torch.argmax(y_probs,dim=1).to(device="cpu")
                y_pred = possible_factors[y_pred_idx]

                # Compute the loss
                loss = nn.functional.cross_entropy(y_logits, y_true_idx.to(device=device,dtype=torch.long))
                acc = torch.eq(y_true, y_pred).sum().item()/ y_true.shape[0] * 100
                total_acc+=acc

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch%1==0:

                    print("Epoch :", epoch+1, "\tLoss :", loss.item(), "\tAccuracy :", acc, "\tAverage accuracy :", total_acc/(epoch+1))
                
                torch.save(classifier.state_dict(), f"./classifier_factor_z_{z}.pt")
        elif mode =="eval":
            B=800
            L=200
            classifier = LinearClassifier(in_features=z).to(device)
            classifier.load_state_dict(torch.load(f"./classifier_factor_z_{z}.pt"))

            # Define a loss function and optimizer
            model = Factor_VAE(latent_size=z).to(device)
            model.load_state_dict(torch.load(f"./factor_vae_model_z_{z}.pt"))
            # Train the classifier on the training data
            classifier.eval()
            model.eval()
            z_diff = torch.zeros((B, z)).to(device)
            possible_factors = torch.tensor([1,3,4,5])
            y_true_idx = torch.ones(4).multinomial(B,replacement=True)
            y_true = possible_factors[y_true_idx] # Choose the fixed parameters
            for b in tqdm(range(B)):
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
            y_logits = classifier(z_diff)
            y_probs = torch.softmax(y_logits, dim=1)
            y_pred_idx = torch.argmax(y_probs,dim=1).to(device="cpu")
            y_pred = possible_factors[y_pred_idx]

            # Compute the loss
            acc = torch.eq(y_true, y_pred).sum().item()/ y_true.shape[0] * 100
            print("Factor, Accuracy :", acc)
        

