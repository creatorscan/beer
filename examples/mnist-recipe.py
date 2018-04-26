# Add "beer" to the PYTHONPATH
import sys
sys.path.insert(0, '../')

import copy

import beer
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

# For plotting.
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, gridplot
from bokeh.models import LinearAxis, Range1d
output_notebook()

# Convenience functions for plotting.
import plotting


root = './data'
download = False  # set to True if the line "train_set = ..." complains

trans = transforms.Compose([
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(), 
#    transforms.Normalize((0.5,), (1.0,)),
])
train_set = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = torchvision.datasets.MNIST(root=root, train=False, transform=trans)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

observed_dim = 28*28
latent_dim = 2

hidden_dim = 400

enc_nn = torch.nn.Sequential(
    torch.nn.Linear(observed_dim, hidden_dim),
    torch.nn.Tanh(),
)
enc_proto = beer.models.MLPNormalDiag(enc_nn, latent_dim)

dec_nn = torch.nn.Sequential(    
    torch.nn.Linear(latent_dim, hidden_dim),
    torch.nn.Tanh(),
)
dec_proto = beer.models.MLPBernoulli(dec_nn, observed_dim)

import copy
latent_normal = beer.models.FixedIsotropicGaussian(latent_dim)
vae = beer.models.VAE(copy.deepcopy(enc_proto), copy.deepcopy(dec_proto), latent_normal, nsamples=1)
mean_elbos = []
mean_klds = []
mean_llhs = []

def train(nb_epochs):
    for i in range(nb_epochs):
        for X, _ in train_loader:
            X = torch.autograd.Variable(X.view(-1, 28**2))
            sth = vae.forward(X)
            neg_elbo, llh, kld = vae.loss(X, sth)
            obj = neg_elbo.mean()
            mean_elbos.append(-obj.item())
            mean_klds.append(kld.mean().item())
            mean_llhs.append(llh.mean().item())
            optim.zero_grad()
            obj.backward()
            optim.step()
        print("epoch {} done, last ELBO: {}".format(i, mean_elbos[-1]))

# a reasonable training procedure
optim = torch.optim.Adam(vae.parameters(), lr=1e-3)
train(1)
