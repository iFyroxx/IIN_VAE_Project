from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch
import numpy as np
import time
import math
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


########### utils ######################
##    This part of the code come from:
##    https://github.com/stfc-sciml/dae
########################################
def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    if not training:
        X_hat = (X - moving_min)/moving_mag
    else:
        assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat =  (X - min_)/mag_
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat*gamma*alpha) + beta
    return Y, moving_mag.data, moving_min.data




class Stretch(nn.Module):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01*torch.ones(shape))
        self.beta = nn.Parameter(np.pi*torch.ones(shape))
        self.register_buffer('moving_mag', 1.*torch.ones(shape))
        self.register_buffer('moving_min', np.pi*torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
        Y, self.moving_mag, self.moving_min = stretch(
            X, self.alpha , self.gamma, self.beta, self.moving_mag, self.moving_min,
            eps=1e-5, momentum=0.99, training = self.training)
        return Y
    
###################################################################
##                         ARCHITECTURE                          ##
###################################################################
   

class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128*2*2, output_dim) #output dim will be z_dim
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


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 128*2*2)
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
        x_recon = self.layers(z)
        return x_recon
    
############### MODEL ##########################
## Adapted from the code of the paper of the DAE
################################################
    
class DAE(nn.Module):
    def __init__(self, z_dim, alpha):
        super(DAE, self).__init__()
        self.latent_dim = z_dim
        self.alpha= alpha
        self.encoder = ConvEncoder(self.latent_dim)
        self.decoder = ConvDecoder(2*self.latent_dim) #*2 because of euler encoding
        self.stretch = Stretch(self.latent_dim, 2, self.alpha)

    def reconstr(self, x):
      """
      to use at inference time
      no gaussian interpolation
      """
      z = self.encoder(x)
      z = self.stretch(z)
      c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 0)
      c = c.T.reshape(self.latent_dim*2, -1).T
      reconstr = self.decoder(c)
      return reconstr

    def reparameterize(self, z):
        """
        perform gaussian interpolation and euler encoding
        """
        diff = torch.abs(z - z.unsqueeze(axis = 1))
        none_zeros = torch.where(diff == 0., torch.tensor([100.]).to(z.device), diff)
        z_scores,_ = torch.min(none_zeros, axis = 1)
        std =  torch.normal(mean = 0., std = 1.*z_scores).to(z.device)
        s = z + std
        c = torch.cat((torch.cos(2*np.pi*s), torch.sin(2*np.pi*s)), 0)
        c = c.T.reshape(self.latent_dim*2,-1).T
        return c

    def forward(self, x):
      """
      to use at training time
      use the gaussian interpolation
      """
      z = self.encoder(x)
      z = self.stretch(z)
      c = self.reparameterize(z) #gaussian interpolation + euler encoding
      reconstr = self.decoder(c)
      return [reconstr, c, z]

    ############# for visualization only ##########################
    def encode(self, x):
      z = self.encoder(x)
      z = self.stretch(z)
      c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 0)
      c = c.T.reshape(self.latent_dim*2, -1).T
      return [c, z]
    
    def decode(self, c):
        samples = self.decoder(c)
        return samples

################################################################
####                   Train the model                        ##  
################################################################


# Dataset initialization
dataset_imgs = torch.tensor(np.load('./data/dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)

torch.manual_seed(10)
imgs_train_set, imgs_test_set, imgs_val_set = torch.utils.data.random_split(dataset_imgs, [0.80, 0.10, 0.10])
batch_size = 64
imgs_train_set = DataLoader(imgs_train_set, batch_size=batch_size, shuffle=True)
imgs_test_set = DataLoader(imgs_test_set, batch_size=batch_size, shuffle=False)
imgs_val_set = DataLoader(imgs_val_set, batch_size=batch_size, shuffle=True)

##### PARAMETERS ##############
z_dim = 4
#vector that we get by doing the PCA on the dataset as described in the article
alpha = torch.Tensor([[1., 1., 0.01, 0.01]]).to(device)
lr = 0.0001
num_epochs = 2
###############################


dae = DAE(z_dim=z_dim, alpha=alpha).cuda()
optimizer = optim.Adam(dae.parameters(), lr=lr)
criterion = nn.BCELoss()

# initialize loss accumulator
train_loss = []
val_loss = []

#training loop
for epoch in range(0, num_epochs):
  loss_tracking = []
  epoch_time = time.time()
  if epoch > 150 and epoch % 50 == 0:
    #save the model
    torch.save(dae.state_dict(), f"./DAE-{epoch}epochs.pt")
  for i, x in enumerate(imgs_train_set):
    dae.train()
    optimizer.zero_grad()
    x = x.cuda()
    x = Variable(x)
    # compute gradient and accumulate loss
    x_recon, _, _= dae(x) #with gaussian interpolation
    loss = criterion(x_recon, x)
    loss_tracking.append(loss.item())
    loss.backward()
    optimizer.step()
  train_loss.append(sum(loss_tracking) / len(loss_tracking))
  print('[epoch %03d] time: %.2f training average loss: %.4f' % (epoch, time.time() - epoch_time, train_loss[-1]))
  for i, x in enumerate(imgs_val_set):
    loss_tracking = []
    dae.eval()
    x = x.cuda()
    x_recon = dae.reconstr(x) #without gaussian interpolation
    loss = criterion(x_recon, x)
    loss_tracking.append(loss.item())
  val_loss.append(sum(loss_tracking) / len(loss_tracking))
  print('[epoch %03d] time: %.2f validation average loss: %.4f' % (epoch, time.time() - epoch_time, val_loss[-1]))

#save the model
torch.save(dae.state_dict(), "./DAE.pt")

#plot the learning curve
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#evaluate the reconstruction on the test dataset
reconstruction_crit = nn.BCELoss()
dae.eval()
bce = []
for i, x in enumerate(imgs_test_set):
    x = x.cuda()
    # reconstruct the image
    x_recon = dae.reconstr(x)
    bce.append(reconstruction_crit(x_recon.view(-1), x.view(-1)))
print(f"the average reconstruction loss on the test dataset is {sum(bce)/len(bce)}")