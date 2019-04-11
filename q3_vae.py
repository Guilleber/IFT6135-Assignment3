import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F
import numpy as np
import argparse
import gc


parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_true", help="Flag to specify if we train the model")
parser.add_argument("--save_path", type=str, default="q2.pt")
parser.add_argument("--load_path", type=str, default="q2.pt")


class View(nn.module):

    def __init__(self, shape):
        super()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):

    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz

        self.enc = nn.Sequential([
            nn.Linear(4*4*512)
        ])

        self.dec = nn.ModuleDict({
            "linear": nn.Linear(self.dimz, 256),
            "conv": nn.Sequential(
                nn.ELU(),
                nn.Conv2d(256, 64, 5, padding=4),
                nn.ELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(64, 32, 3, padding=2),
                nn.ELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(32, 16, 3, padding=2),
                nn.ELU(),
                nn.Conv2d(16, 1, 3, padding=2)
            )
        }
        )

    def forward(self, x):
        x = self.enc["conv"](x)
        q_params = self.enc["linear"](x.view(-1, 256))
        mu, log_sigma = q_params[:, :self.dimz], q_params[:, self.dimz:]

        sigma = torch.exp(log_sigma) + 1e-7
        e = torch.randn_like(mu, dtype=torch.float32)
        z = mu + sigma * e
        x_ = self.dec["linear"](z)
        x_ = self.dec["conv"](x_.view(-1, 256, 1, 1))

        return mu, log_sigma, torch.squeeze(x_, dim=1)

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


def recon_loss(x, x_):
    """
    Function that computes the reconstruction loss between the input and the generated samples
    :param x: Input of size [batch_size, 784]
    :param x_: Generated samples of size [batch_size, 784]
    :return: The reconstruction loss for each batch item. Size is [batch_size,]
    """
    return F.binary_cross_entropy_with_logits(x_, x, reduction="none").sum(dim=-1)



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
        for i in range(int(np.ceil(train.shape[0] / model.batch_size))):

            batch = train[model.batch_size*i:model.batch_size*(i+1)]

            # obtain the parameters from the encoder and compute KL divergence
            mu, log_sigma, x_ = model(batch)
            kl = kl_div(mu, log_sigma)

            # compute the reconstruction loss
            logpx_z = - recon_loss(batch.view([-1, 784]), x_.view([-1, 784]))

            # combine the two loss terms and compute gradients
            elbo = (logpx_z - kl).mean()

            # maximize the elbo i.e. minimize - elbo
            autograd.backward([-elbo])

            # Update the parameters and zero the gradients for the next mini-batch
            adam.step()
            adam.zero_grad()

        # compute the loss for the validation set
        mu, log_sigma, x_ = model(valid)
        kl = kl_div(mu, log_sigma)
        logpx_z = - recon_loss(valid.view([-1, 784]), x_.view([-1, 784]))
        elbo = (logpx_z - kl).mean()
        print("After epoch {} the validation loss is: ".format(epoch+1), elbo.item())

    # save the model to be used later
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    # get the arguments
    args = parser.parse_args()

    # load the dataset
    train = torch.from_numpy(np.loadtxt("binarized_mnist_train.amat").astype(np.float32))
    valid = torch.from_numpy(np.loadtxt("binarized_mnist_valid.amat").astype(np.float32))
    test = torch.from_numpy(np.loadtxt("binarized_mnist_test.amat").astype(np.float32))

    # reshape the data for usage in the model
    train = train.view([-1, 1, 28, 28])
    valid = valid.view([-1, 1, 28, 28])
    test = test.view([-1, 1, 28, 28])

    # Create model. Load or train depending on choice
    model = VAE(batch_size=1, dimz=100)
    if parser.parse_args().t:
        train_model(model, train, valid, args.save_path)
    else:
        model.load_state_dict(torch.load(args.load_path))
        model.eval()

    # Evaluate part
