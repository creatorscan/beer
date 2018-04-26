# Add "beer" to the PYTHONPATH
import sys
sys.path.insert(0, '../')

import copy

import argparse

import beer
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms


class BatchNumberLimiter:
    ''' A wrapper for any data source.
        Limits the number of batches that go out.
    '''
    def __init__(self, source, limit):
        self._source = source
        self._limit = limit

    def __iter__(self):
        batches_out = 0
        for b in self._source:
            yield b
            batches_out += 1
            if batches_out > self._limit:
                raise StopIteration

    def __len__(self):
        return self._limit


class SimpleLogger:
    def __init__(self, report_interval):
        self._mean_elbos = []
        self._mean_klds = []
        self._mean_llhs = []
        self._report_interval = report_interval

    def log(self, total_loss):
        self._mean_elbos.append(-total_loss[0].mean().item()) 
        self._mean_llhs.append(total_loss[1].mean().item())
        self._mean_klds.append(total_loss[2].mean().item())

        if len(self._mean_elbos) > 0 and len(self._mean_elbos) % self._report_interval == 0:
            reported_suffix = self._mean_elbos[-self._report_interval:]
            print(sum(reported_suffix) / len(reported_suffix))


def train(vae, data, optim, loss_logger=None):
    for X, _ in data:
        X = torch.autograd.Variable(X.view(-1, 28**2)).to(device)
        sth = vae.forward(X)
        complete_loss = vae.loss(X, sth)
        obj = complete_loss[0].mean()
        optim.zero_grad()
        obj.backward()
        optim.step()

        if loss_logger is not None:
            loss_logger.log(complete_loss)


def evaluate(vae, data):
    elbos = []
    for X, _ in data:
        X = torch.autograd.Variable(X.view(-1, 28**2)).to(device)
        sth = vae.forward(X)
        complete_loss = vae.loss(X, sth)
        obj = complete_loss[0].mean()
        elbos.append(-obj.item())

    return sum(elbos) / len(elbos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16,
        help="batch size")
    parser.add_argument("--latent-dim", type=int, default=2,
        help="dimensionality of the latent space (where the KLD is computed)")
    parser.add_argument("--nb-hidden", type=int, default=100,
        help="dimensionality of hidden layers of encoder/decoder")
    # this is NOT (yet) what Lucas asked for for the NIPS experiments,
    # as data is, in every epoch, first shuffled and then cut
    parser.add_argument("--nb-batches", type=int, default=60001,
        help="how many training batches to supply. Effectively restricts training data.")
    parser.add_argument("--nb-epochs", type=int, default=1,
        help="number of runs through training data")
    parser.add_argument("--cuda", action='store_true',
        help="run on CUDA")
    parser.add_argument("--report-interval", type=int, default=50,
        help="how often to report ELBO")
    args = parser.parse_args()

    root = './data'
    download = False  # set to True if the line "train_set = ..." complains

    device = torch.device("cuda:0" if args.cuda else "cpu")

    trans = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(), 
    ])
    train_set = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = torchvision.datasets.MNIST(root=root, train=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=args.batch_size,
                     shuffle=True)
    train_loader = BatchNumberLimiter(train_loader, args.nb_batches)

    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=args.batch_size,
                    shuffle=False)

    observed_dim = 28*28

    enc_nn = torch.nn.Sequential(
        torch.nn.Linear(observed_dim, args.nb_hidden),
        torch.nn.Tanh(),
    )
    enc_proto = beer.models.MLPNormalDiag(enc_nn, args.latent_dim)

    dec_nn = torch.nn.Sequential(    
        torch.nn.Linear(args.latent_dim, args.nb_hidden),
        torch.nn.Tanh(),
    )
    dec_proto = beer.models.MLPBernoulli(dec_nn, observed_dim)

    latent_normal = beer.models.FixedIsotropicGaussian(args.latent_dim)
    vae = beer.models.VAE(copy.deepcopy(enc_proto), copy.deepcopy(dec_proto), latent_normal, nsamples=1)
    vae.to(device)

    logger = SimpleLogger(args.report_interval)
    optim = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch_no in range(1, args.nb_epochs+1):
        train(vae, train_loader, optim, logger)

    print("Test mean ELBO:", evaluate(vae, test_loader))
