import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F
from torch import cuda
from classify_svhn import get_data_loader
import numpy as np
import argparse
import gc


parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_true", help="Flag to specify if we train the model")
parser.add_argument("--save_path", type=str, default="q3_vae.pt")
parser.add_argument("--load_path", type=str, default="q3_vae.pt")
parser.add_argument("--batch_size", type=int, default=32, help="Size of the mini-batches")
parser.add_argument("--dimz", type=int, default=100, help="Dimension of the latent variables")
parser.add_argument("--data_path", type=str, default="svhn.mat", help="SVHN dataset location")

# get the arguments
args = parser.parse_args()
args.device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')


class View(nn.Module):

    def __init__(self, shape, *shape_):
        super().__init__()
        if isinstance(shape, list):
            self.shape = shape
        else:
            self.shape = (shape,) + shape_

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):

    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz

        self.enc = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(2e-2),

            #  Layer 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(2e-2),

            # Layer 3
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(2e-2),

            # Layer 4
            View(self.batch_size, 4*4*512),
            nn.Linear(4*4*512, 2*self.dimz)
        )

        self.dec = nn.Sequential(
            # Layer 1
            nn.Linear(self.dimz, 4*4*512),
            View(self.batch_size, 512, 4, 4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(2e-2),

            # Layer 2
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(2e-2),

            # Layer 3
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(2e-2),

            # Layer 4
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(2e-2)
        )

    def forward(self, x):
        """
        Forward pass for the network
        :param x: The input image of size [batch_size, 3, 32, 32]
        :return:
        """
        # get the encoder output
        enc = self.enc(x)
        mu, log_sigma = enc[:, :self.dimz], enc[:, self.dimz:]

        # get the sample z
        z = torch.randn_like(mu,  device=args.device)

        # sample from qz_x
        f_x = mu + torch.exp(log_sigma) * z

        # get decoder output
        g_z = self.dec(f_x)

        return mu, log_sigma, g_z

    def generate(self, z):
        """
        Generate multiple samples
        :param z: Size is [batch_size, n_samples, dimz]
        :return:
        """
        # To do
        pass


def kl_div(mu, log_sigma):
    """
    Function that computes the KL divergence
    :param mu: Mean parameter of size [batch_size, dimz]
    :param log_sigma: Covariance parameter (diagonal of matrix) of size [batch_size, dimz]
    :return: The KL divergence for each element in the batch. Size is [batch_size,]
    """
    return 0.5 * (-1. - 2.*log_sigma + torch.exp(log_sigma)**2. + mu**2.).sum(dim=1)


def ll(x, x_):
    """
    Function that computes the log-likelihood of p(x|z)
    :param x: Input of size [batch_size, 3*32*32]
    :param x_: Generated samples of size [batch_size, 3*32*32]
    :return:
    """
    k = x.size()[1]
    return -k/2 * torch.log(2 * np.pi * torch.ones(1)) -0.5 * ((x - x_)**2.).mean(dim=1)


def train_model(model, train, valid, save_path):
    """
    Function that trains the model
    :param model: The model to train
    :param train: The training set
    :param valid: The validation set
    :return:
    """
    # optimizer for the network
    adam = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(20):
        for batch, i in train:
            # put batch on device
            batch = batch.to(args.device)

            # obtain the parameters from the encoder and compute KL divergence
            mu, log_sigma, g_z = model(batch)
            kl = kl_div(mu, log_sigma)

            # compute the reconstruction loss
            logpx_z = ll(batch.view(-1, 3*32*32), g_z.view(-1, 3*32*32))

            # combine the two loss terms and compute gradients
            elbo = (logpx_z - kl).mean()

            # maximize the elbo i.e. minimize - elbo
            autograd.backward([-elbo])

            # Update the parameters and zero the gradients for the next mini-batch
            adam.step()
            adam.zero_grad()

        # compute the loss for the validation set
        valid_elbo = torch.zeros(1)
        nb_batches = 0
        for batch, i in valid:
            nb_batches += 1
            batch = batch.to(args.device)
            mu, log_sigma, g_z = model(batch)
            kl = kl_div(mu, log_sigma)
            logpx_z = ll(valid.view(-1, 3*32*32), g_z.view(-1, 3*32*32))
            valid_elbo += (logpx_z - kl).mean()
        valid_elbo /= nb_batches
        print("After epoch {} the validation loss is: ".format(epoch+1), valid_elbo.item())

    # save the model to be used later
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    # check for cuda
    device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')
    args.device = device

    # load the dataset
    train, valid, test = get_data_loader(args.data_path, args.batch_size)

    # Create model. Load or train depending on choice
    model = VAE(batch_size=args.batch_size, dimz=args.dimz)
    if parser.parse_args().t:
        train_model(model, train, valid, args.save_path)
    else:
        model.load_state_dict(torch.load(args.load_path))
        model.eval()

    # Evaluate part
