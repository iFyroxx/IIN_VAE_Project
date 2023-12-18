import torch
import torch.nn as nn
from "./beta-VAE" import beta_VAE

class LinearClassifier(nn.Module):
    def __init__(self, in_features = 5, out_features = 1):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

classifier = LinearClassifier()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

B=64
L=64
device = "cuda"
model = beta_VAE(latent_size=5).to(device)
# model.load_state_dict(torch.load("./beta4_vae.pt"))
# Train the classifier on the training data
for epoch in range(100):
    z_diff = torch.zeros((B, 5)).to(device)
    y = torch.randint(5, (B,))
    for b in range(B):
        z_diff_b = torch.zeros(1,5).to(device)
        for l in range(L):
            v1 = torch.randint(len(labels),(1,))
            v2 = torch.randint(len(labels),(1,))
            while v2==v1 or labels[v2][y[b]]!=labels[v1][y[b]]:
                v2 = torch.randint(len(labels),(1,))
            x1 = images[v1].to(device)
            x2 = images[v2].to(device)  
            latent1 = model.encoder(x1).detach()
            latent2 = model.encoder(x2).detach()
            z1 = latent1[:, :5]
            z2 =latent2[:, :5]
            z_diff_b += torch.abs(z1-z2)
        z_diff_b = z_diff_b/L
        z_diff[b] = z_diff_b
    # Get the next batch of training data
    x_train, y_train = 

    # Forward pass
    y_pred = classifier(x_train)

    # Compute the loss
    loss = criterion(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()