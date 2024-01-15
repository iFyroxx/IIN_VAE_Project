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


############# UTILS #####################
#  this part of the code come from:
#  https://github.com/rtqichen/beta-tcvae
#########################################
class Normal(nn.Module):
    """Samples from a Normal distribution using the reparameterization trick.
    """

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        self.normalization = Variable(torch.Tensor([np.log(2 * np.pi)]))

        self.mu = Variable(torch.Tensor([mu]))
        self.logsigma = Variable(torch.Tensor([math.log(sigma)]))

    def _check_inputs(self, size, mu_logsigma):
        if size is None and mu_logsigma is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0).expand(size)
            logsigma = mu_logsigma.select(-1, 1).expand(size)
            return mu, logsigma
        elif size is not None:
            mu = self.mu.expand(size)
            logsigma = self.logsigma.expand(size)
            return mu, logsigma
        elif mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0)
            logsigma = mu_logsigma.select(-1, 1)
            return mu, logsigma
        else:
            raise ValueError(
                'Given invalid inputs: size={}, mu_logsigma={})'.format(
                    size, mu_logsigma))

    def sample(self, size=None, params=None):
        mu, logsigma = self._check_inputs(size, params)
        std_z = Variable(torch.randn(mu.size()).type_as(mu.data))
        sample = std_z * torch.exp(logsigma) + mu
        return sample

    def log_density(self, sample, params=None):
        if params is not None:
            mu, logsigma = self._check_inputs(None, params)
        else:
            mu, logsigma = self._check_inputs(sample.size(), None)
            mu = mu.type_as(sample)
            logsigma = logsigma.type_as(sample)

        c = self.normalization.type_as(sample.data)
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logsigma + c)

    def NLL(self, params, sample_params=None):
        """Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.
        """
        mu, logsigma = self._check_inputs(None, params)
        if sample_params is not None:
            sample_mu, sample_logsigma = self._check_inputs(None, sample_params)
        else:
            sample_mu, sample_logsigma = mu, logsigma

        c = self.normalization.type_as(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
            + torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)

    def kld(self, params):
        """Computes KL(q||p) where q is the given distribution and p
        is the standard Normal distribution.
        """
        mu, logsigma = self._check_inputs(None, params)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
        kld.mul_(-0.5)
        return kld

    def get_params(self):
        return torch.cat([self.mu, self.logsigma])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(
            self.mu.data[0], self.logsigma.exp().data[0])
        return tmpstr



class Bernoulli(nn.Module):
    """Samples from a Bernoulli distribution where the probability is given
    by the sigmoid of the given parameter.
    """

    def __init__(self, p=0.5, stgradient=False, eps = 1e-8
):
        super(Bernoulli, self).__init__()
        p = torch.Tensor([p])
        self.p = Variable(torch.log(p / (1 - p) + eps))
        self.stgradient = stgradient

    def _check_inputs(self, size, ps):
        if size is None and ps is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and ps is not None:
            if ps.ndimension() > len(size):
                return ps.squeeze(-1).expand(size)
            else:
                return ps.expand(size)
        elif size is not None:
            return self.p.expand(size)
        elif ps is not None:
            return ps
        else:
            raise ValueError(
                'Given invalid inputs: size={}, ps={})'.format(size, ps))

    def _sample_logistic(self, size, eps=1e-8):
        u = Variable(torch.rand(size))
        l = torch.log(u + eps) - torch.log(1 - u + eps)
        return l

    def sample(self, size=None, params=None):
        presigm_ps = self._check_inputs(size, params)
        logp = F.logsigmoid(presigm_ps)
        logq = F.logsigmoid(-presigm_ps)
        l = self._sample_logistic(logp.size()).type_as(presigm_ps)
        z = logp - logq + l
        b = STHeaviside.apply(z)
        return b if self.stgradient else b.detach()

    def log_density(self, sample, params=None, eps=1e-8):
        presigm_ps = self._check_inputs(sample.size(), params).type_as(sample)
        p = (F.sigmoid(presigm_ps) + eps) * (1 - 2 * eps)
        logp = sample * torch.log(p + eps) + (1 - sample) * torch.log(1 - p + eps)
        return logp

    def get_params(self):
        return self.p

    @property
    def nparams(self):
        return 1

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return self.stgradient

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f})'.format(
            torch.sigmoid(self.p.data)[0])
        return tmpstr

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class STHeaviside(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(x.size()).type_as(x)
        y[x >= 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
 
########################################################
##                ARCHITECTURE                        ##
########################################################
class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128*2*2, output_dim) #output dim will be 2*z_dim
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
        )

    def forward(self, z):
        mu_img = self.layers(z)
        return mu_img
    
################### MODEL #######################
##   Adapted from the original implementation     
##      of the TC-VAE paper
#################################################
    
class VAE(nn.Module):
    def __init__(self, z_dim, beta, use_cuda=True):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.beta = beta #penalisation
        self.x_dist = Bernoulli()

        #distribution
        self.prior_dist = Normal()
        self.q_dist = Normal()
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

       
        self.encoder = ConvEncoder(z_dim * self.q_dist.nparams) #to return mu and sigma
        self.decoder = ConvDecoder(z_dim)
        if use_cuda:
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        x_params = self.decoder.forward(zs)
        return x_params

    #q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    #p(x|z)
    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    #simplified version of the elbo computation given in the TC-VAE code
    def penalised_elbo(self, x, dataset_size):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        #elbo = logpx + logpz - logqz_condx thus beta-VAE require only the previous lines

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )
        # minibatch weighted sampling
        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))

        penalised_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (logqz_prodmarginals - logpz)
        return penalised_elbo
    
############## Train the Model ####################
##### PARAMETERS ##############
z_dim = 4
beta = 6
lr = 0.0001
num_epochs = 500
###############################
# Dataset initialization
dataset_imgs = torch.tensor(np.load('./data/dsprites_no_scale.npz', allow_pickle=True, encoding='bytes')["imgs"], dtype=torch.float).unsqueeze(1)

torch.manual_seed(10)
imgs_train_set, imgs_test_set, imgs_val_set = torch.utils.data.random_split(dataset_imgs, [0.80, 0.10, 0.10])
batch_size = 64
imgs_train_set = DataLoader(imgs_train_set, batch_size=batch_size, shuffle=True)
imgs_test_set = DataLoader(imgs_test_set, batch_size=batch_size, shuffle=False)
imgs_val_set = DataLoader(imgs_val_set, batch_size=batch_size, shuffle=True)


vae = VAE(z_dim=z_dim, beta=beta)
optimizer = optim.Adam(vae.parameters(), lr=lr)
train_dataset_size = len(imgs_train_set.dataset)
val_dataset_size = len(imgs_val_set.dataset)
train_dataset_size = len(imgs_train_set.dataset)
test_dataset_size = len(imgs_test_set.dataset)


train_loss = []
val_loss = []

#training loop
for epoch in range(0, num_epochs):
  elbo_tracking = []
  epoch_time = time.time()
  if epoch > 150 and epoch % 50 == 0:
    #save the model
    torch.save(vae.state_dict(), f"./TCVAE-{epoch}epochs.pt")
  for i, x in enumerate(imgs_train_set):
    vae.train()
    optimizer.zero_grad()
    x = x.cuda()
    x = Variable(x)
    # do ELBO gradient and accumulate loss
    penalised_elbo = vae.penalised_elbo(x, train_dataset_size)
    penalised_elbo.mean().mul(-1).backward()
    elbo_tracking.append(penalised_elbo.mean().mul(-1).item())
    optimizer.step()
  train_loss.append(sum(elbo_tracking) / len(elbo_tracking))
  print('[epoch %03d] time: %.2f training average loss: %.4f' % (epoch, time.time() - epoch_time, train_loss[-1]))
  for i, x in enumerate(imgs_val_set):
    elbo_tracking = []
    vae.eval()
    x = x.cuda()
    # compute ELBO
    penalised_elbo = vae.penalised_elbo(x, val_dataset_size)
    penalised_elbo.mean().mul(-1)
    elbo_tracking.append(penalised_elbo.mean().mul(-1).item())
  val_loss.append(sum(elbo_tracking) / len(elbo_tracking))
  print('[epoch %03d] time: %.2f validation average loss: %.4f' % (epoch, time.time() - epoch_time, val_loss[-1]))

#save the model
torch.save(vae.state_dict(), "./TC-VAE.pt")

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
vae.eval()
bce = []
for i, x in enumerate(imgs_test_set):
    # transfer to GPU
    x = x.cuda()
    # reconstruct the image
    x_recon, _, _, _ = vae.reconstruct_img(x)
    bce.append(reconstruction_crit(x_recon.view(-1), x.view(-1)))
print(f"the average reconstruction loss on the test dataset is {sum(bce)/len(bce)}")

