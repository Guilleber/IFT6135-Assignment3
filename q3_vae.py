import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F
from torch import cuda
from torchvision import transforms
from classify_svhn import get_data_loader
import numpy as np
import argparse
import os


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
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(2e-1),

            #  Layer 2
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(2e-1),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(2e-1),

            # Layer 4
            View(-1, 4*4*256),
            nn.Linear(4*4*256, 2 * self.dimz)
        )

        self.dec = nn.Sequential(
            # Layer 1
            nn.Linear(self.dimz, 4*4*512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(),
            View(-1, 512, 4, 4),

            # Layer 2
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 3
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
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
    return -k/2 * torch.log(2 * np.pi * torch.ones(1, device=args.device)) -0.5 * ((x - x_)**2.).sum(dim=1)


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

    for epoch in range(args.nb_epochs):
        for i, (batch, label) in enumerate(train):
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
        with torch.no_grad():
            valid_elbo = torch.zeros(1)
            nb_batches = 0
            for i, (batch, label) in enumerate(valid):
                nb_batches += 1
                batch = batch.to(args.device)
                mu, log_sigma, g_z = model(batch)
                kl = kl_div(mu, log_sigma)
                logpx_z = ll(batch.view(-1, 3*32*32), g_z.view(-1, 3*32*32))
                valid_elbo += (logpx_z - kl).mean()
            valid_elbo /= nb_batches
            print("After epoch {} the validation loss is: ".format(epoch+1), valid_elbo.item())

    # save the model to be used later
    torch.save(model.state_dict(), save_path)


def evaluation(model):
    """
    Function that generates samples for the qualitative evaluation of the model
    :param model: The model from which we pull samples
    :return:
    """
    with torch.no_grad():
        transf = transforms.ToPILImage()
        z = torch.rand(model.batch_size, model.dimz, device=args.device)
        samples = model.dec(z)

        if not os.path.isdir(args.sample_dir):
            os.mkdir(args.sample_dir)

        decoder_dir = os.path.join(args.sample_dir, "decoder_samples")
        if not os.path.isdir(decoder_dir):
            os.mkdir(decoder_dir)

        # save the decoder samples
        for i, sample in enumerate(samples):
            im = transf(sample.to(device='cpu'))
            im.save(os.path.join(decoder_dir, "img_{}.jpeg".format(i)))

        # perturb the z and get samples
        z_ = z + args.eps
        samples = model.dec(z_)

        perturb_dir = os.path.join(args.sample_dir, "perturbed_samples")
        if not os.path.isdir(perturb_dir):
            os.mkdir(perturb_dir)

        # save the perturbed samples
        for i, sample in enumerate(samples):
            im = transf(sample.to(device='cpu'))
            im.save(os.path.join(perturb_dir, "p_img_{}.jpeg".format(i)))

        int_dir = os.path.join(args.sample_dir, "interpolated_samples")
        if not os.path.isdir(int_dir):
            os.mkdir(int_dir)

        # interpolate between two z's and generate samples. Save them
        for a in range(11):
            z_a = a / 10. * z[0:1] + (1. - a/10.) * z[1:2]
            gz_a = model.dec(z_a)[0]
            im = transf(gz_a.to(device='cpu'))
            im.save(os.path.join(int_dir, "i1_img_{}.jpeg".format(a)))

        # interpolate the result of the two g(z) values
        g_z = model.dec(z[0:2])
        for a in range(11):
            x_a = a / 10. * g_z[0] + (1. - a / 10.) * g_z[1]
            im = transf(x_a.to(device='cpu'))
            im.save(os.path.join(int_dir, "i2_img_{}.jpeg".format(a)))

        # sample 1000 images to use for FID score
        thousand_dir = os.path.join(args.sample_dir, "1000_samples", "samples")
        if not os.path.isdir(thousand_dir):
            os.mkdir(thousand_dir)

        z = torch.randn(1000, model.dimz)
        gz = model.deconvs(z)
        for i, sample in enumerate(gz):
            im = transf(sample.to(device='cpu'))
            im.save(os.path.join(thousand_dir, "img_{}.jpeg".format(i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store_true", help="Flag to specify if we train the model")
    parser.add_argument("--save_path", type=str, default="q3_vae.pt")
    parser.add_argument("--load_path", type=str, default="q3_vae.pt")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the mini-batches")
    parser.add_argument("--dimz", type=int, default=100, help="Dimension of the latent variables")
    parser.add_argument("--data_dir", type=str, default="svhn.mat", help="SVHN dataset location")
    parser.add_argument("--nb_epochs", type=int, default=50, help="The number of epochs for training")
    parser.add_argument("--eps", type=float, default=1e-1, help="Perturbation value to the latent when evaluating")
    parser.add_argument("--sample_dir", type=str, default="samples", help="Directory containing samples for"
                                                                          "evaluation")

    # get the arguments
    args = parser.parse_args()
    args.device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')
    # check for cuda
    device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')
    args.device = device

    # load the dataset
    train, valid, test = get_data_loader(args.data_dir, args.batch_size)

    # Create model. Load or train depending on choice
    model = VAE(batch_size=args.batch_size, dimz=args.dimz).to(args.device)
    if args.t:
        train_model(model, train, valid, args.save_path)
    else:
        model.load_state_dict(torch.load(args.load_path))
        model.eval()
        evaluation(model)
