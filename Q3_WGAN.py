import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch import optim
from torch import autograd
from classify_svhn import get_data_loader
from q3_vae import View
import numpy as np
import matplotlib.pyplot as plt
import samplers
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_true", help="Flag to specify if we train the model")
parser.add_argument("--save_path", type=str, default="q3_gan.pt")
parser.add_argument("--load_path", type=str, default="q3_gan.pt")
parser.add_argument("--batch_size", type=int, default=32, help="Size of the mini-batches")
parser.add_argument("--dimz", type=int, default=100, help="Dimension of the latent variables")
parser.add_argument("--data_path", type=str, default="svhn.mat", help="SVHN dataset location")
parser.add_argument("--lam",  type=int, default=20, help="Lambda coefficient for the regularizer in"
                                                            "in the WGAN-GP loss")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the optimzer")
parser.add_argument("--update_ratio", type=int, default=20, help="The number of updates to  the discriminator"
                                                                 "before one update to the generator")

# get the arguments
args = parser.parse_args()
args.device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')


class D(nn.Module):

    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz

        self.convs = nn.Sequential(
            # layer1
            nn.Conv2d(3, 128, 4, padding=1, stride=2),
            nn.LeakyReLU(2e-2),

            # layer2
            nn.Conv2d(128, 256, 4, padding=1, stride=2),
            nn.LeakyReLU(2e-2),

            # layer3
            nn.Conv2d(256, 512, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),

            # layer 4
            View(self.batch_size, 4*4*512),
            nn.Linear(4 * 4 * 512, self.dimz),
            nn.LeakyReLU(2e-2),

            # layer 5
            nn.Linear(self.dimz, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.convs(x)
        return out
    

class G(nn.Module):

    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz
        
        self.deconvs = nn.Sequential(
            # layer 1
            nn.Linear(self.dimz, 4 * 4 * 512),
            nn.LeakyReLU(2e-2),
            View(self.batch_size, 512, 4, 4),

            # layer2
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),
            nn.LeakyReLU(2e-2),

            # layer 3
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),
            nn.LeakyReLU(2e-2),

            # layer 4
            nn.ConvTranspose2d(128, 3, 4, padding=1, stride=2),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.deconvs(z)
        return out


def wgan_gp_loss(real, fake, grad, lam):
    """
    Function that computes the WGAN-GP metric given the discriminator's output on real and fake data
    :param real: The output of the discriminator on real data of size [batch_size,]
    :param fake: The output of the generator on fake data of size [batch_size,]
    :param grad: The gradient of the output of the discriminator. Size is [batch_size, 3*32*32]
    :param: The lambda factor applied for regularization
    :return: The WGAN-GP loss over all elements in the mini-batch. Size is [batch_size,]
    """
    return real - fake - lam * (torch.sqrt((grad**2.).sum(dim=1)) - 1)**2


def train_model(g, d, train, valid, save_path):
    """
    Function that trains the model
    :param g: The model generator to train
    :param d: The discriminator to train
    :param train: The training set
    :param valid: The validation set
    :return:
    """
    # optimizer for the network
    g_optim = optim.Adam(g.parameters(), lr=3e-4)
    d_optim = optim.Adam(d.parameters(), lr=3e-4)

    # variable to remember whose turn it is to update its parameters
    turn = 0

    for epoch in range(20):
        for batch, i in train:
            # put batch on device
            batch = batch.to(args.device)

            # obtain the discriminator output on real data
            real_prob = d(batch)

            # obtain the discriminator output on the fake data
            z = torch.randn(g.batch_size, g.dimz, device=args.device)
            fake = g(z)
            fake_prob = d(fake)

            # obtain the gradient term of the WGAN-GP loss
            a = torch.rand_like(batch, device=args.device)
            conv = a * batch + (1 - a) * fake
            d_conv = d(conv)
            grad = autograd.grad(d_conv, conv, torch.ones_like(d_conv),
                                 retain_graph=True, create_graph=True, only_inputs=True)[0]

            # compute the WGAN-GP loss
            loss = wgan_gp_loss(real_prob, fake_prob, grad.view(-1, 3*32*32), args.lam).mean()

            # minimize the loss
            autograd.backward([loss])

            # Update the parameters and zero the gradients for the next mini-batch
            if turn // args.update_ratio == 0:
                d_optim.step()
                d_optim.zero_grad()
                turn += 1
            else:
                g_optim.step()
                g_optim.zero_grad()
                turn = 0

        # compute the loss for the validation set
        valid_loss = torch.zeros(1)
        nb_batches = 0
        for batch, i in valid:
            nb_batches += 1
            batch = batch.to(args.device)
            real_prob = d(batch)
            z = torch.randn(g.batch_size, g.dimz, device=args.device)
            fake = g(z)
            fake_prob = d(fake)
            a = torch.randn_like(batch, device=args.device)
            conv = a * batch + (1. - a) * fake
            d_conv = d(g(conv))
            grad = autograd.grad(d_conv, conv, torch.ones_like(d_conv),
                                 retain_graph=True, create_graph=True, only_inputs=True)[0]
            batch_loss = wgan_gp_loss(real_prob, fake_prob, grad.view(-1, 3*32*32), args.lam)
            valid_loss += batch_loss
        valid_loss /= nb_batches
        print("After epoch {} the validation loss is: ".format(epoch+1), valid_loss.item())

    # save the model to be used later
    torch.save({"generator": g.state_dict(), "discriminator": d}, save_path)


if __name__ == "__main__":
    # check for cuda
    device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')
    args.device = device

    # load the dataset
    train, valid, test = get_data_loader(args.data_path, args.batch_size)

    # Create model. Load or train depending on choice
    g = G(args.batch_size, args.dimz)
    d = D(args.batch_size, args.dimz)
    if parser.parse_args().t:
        train_model(g, d, train, valid, args.save_path)
    else:
        state_dict = torch.load(args.load_path)
        g.load_state_dict(state_dict["generator"])
        g.eval()
        d.load_state_dict(state_dict["discriminator"])
        d.eval()

    # Evaluate part
