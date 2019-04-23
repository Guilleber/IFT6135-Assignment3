import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F
from torch import cuda
import numpy as np
import argparse
import gc


parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_true", help="Flag to specify if we train the model")
parser.add_argument("--save_path", type=str, default="q2.pt")
parser.add_argument("--load_path", type=str, default="q2.pt")

# get the arguments
args = parser.parse_args()
args.device = torch.device('cuda') if cuda.is_available() else torch.device("cpu")


class VAE(nn.Module):

    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz

        self.enc = nn.ModuleDict({
            "conv":nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ELU(),
                nn.AvgPool2d(2, 2),
                nn.Conv2d(32, 64, 3),
                nn.ELU(),
                nn.AvgPool2d(2, 2),
                nn.Conv2d(64, 256, 5),
                nn.ELU()
            ),
            "linear": nn.Linear(256, 2*self.dimz)
        }
        )

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
        e = torch.randn_like(mu, dtype=torch.float32, device=args.device)
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
        x_ = self.dec["linear"](z)
        x_ = self.dec["conv"](x_.view(-1, 256, 1, 1))

        return torch.squeeze(x_, dim=1)

    def params(self, x):
        x = self.enc["conv"](x)
        q_params = self.enc["linear"](x.view(-1, 256))
        mu, log_sigma = q_params[:, :self.dimz], q_params[:, self.dimz:]
        return mu, log_sigma


def kl_div(mu, log_sigma):
    """
    Function that computes the KL divergence
    :param mu: Mean parameter of size [batch_size, dimz]
    :param log_sigma: Log of the sigma value for each dimension of size [batch_size, dimz]
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


def pdf(x, mu, log_sigma):
    """
    Function that computes the log_pdf of x
    :param x: Input of size [batch_size, n_samples, dimz]
    :param mu: Mean of size [batch_size, dimz]
    :param log_sigma: Log  of sigma vector (elementwise) of size [batch_size, dimz]
    :return: The log_pdf for each input of size [batch_size, n_samples]
    """
    sigma = torch.exp(log_sigma) + 1e-7
    k = mu.size()[1]
    mu = mu.view([-1, 1, k])
    sigma = sigma.view([-1, 1, k])
    log_det = log_sigma.sum(dim=-1).view(-1, 1)
    log_exp = -0.5 * ((1./sigma**2.) * (x - mu)**2.).sum(dim=-1)
    return -(k/2.)*np.log(2 * np.pi) - log_det + log_exp


def log_ll(model, x, z):
    """
    Function that computes the log-likelihood estimate for multiple inputs
    :param model: The trained model
    :param x: The input of size [batch_size, dimz]
    :param z: The random samples of size [batch_size, num_samples, dimz]
    :return: The log-likelihood estimates for the multiple points
    """
    with torch.no_grad():
        mu, log_sigma = model.params(x.view([-1, 1, 28, 28]))
        x_ = (model.generate(z.view(-1, model.dimz))).view(model.batch_size, -1, 784)
        logpx_z = torch.stack([-recon_loss(x, x_[:, i, :]) for i in range(z.size()[1])], dim=1)
    logqz_x = pdf(z, mu, log_sigma)
    logpz = pdf(z, torch.zeros_like(mu), torch.ones_like(log_sigma))
    scale = torch.max(logpx_z + logpz - logqz_x, dim=1, keepdim=True)[0]
    return scale.view(-1) + torch.log(torch.exp(logpx_z + logpz - logqz_x - scale).mean(dim=-1))


def evaluate(model, x, z):
    """
    Function that computes the average log_likelihood over all the datapoints of a model
    :param model:
    :param x: Input of size [batch_size, dimz]
    :param z: Samples from N(0,I) of size [batch_size, num_samples, dimz
    :return: The log-likelihood estimate for the dataset
    """
    ll = torch.zeros([model.batch_size,], device=args.device)
    nb_batches = int(np.floor(x.size()[0] / model.batch_size))

    for i in range(nb_batches):
        batch = x[i*model.batch_size:(i+1)*model.batch_size]
        ll += log_ll(model, batch, z)
        gc.collect()
    return ll / nb_batches


if __name__ == "__main__":
    # load the dataset
    train = torch.from_numpy(np.loadtxt("binarized_mnist_train.amat").astype(np.float32)).to(args.device)
    valid = torch.from_numpy(np.loadtxt("binarized_mnist_valid.amat").astype(np.float32)).to(args.device)
    test = torch.from_numpy(np.loadtxt("binarized_mnist_test.amat").astype(np.float32)).to(args.device)

    # reshape the data for usage in the model
    train = train.view([-1, 1, 28, 28])
    valid = valid.view([-1, 1, 28, 28])
    test = test.view([-1, 1, 28, 28])

    # Create model. Load or train depending on choice
    model = VAE(batch_size=32, dimz=100).to(args.device)
    if args.t:
        train_model(model, train, valid, args.save_path)
    else:
        model.load_state_dict(torch.load(args.load_path))
        model.eval()

    z = torch.randn(size=[model.batch_size, 200, 100], device=args.device)
    # compute the log_likelihood for the validation set and the test set
    valid_ll = evaluate(model, valid.view([-1, 784]), z).mean().item()
    gc.collect()
    test_ll = evaluate(model, test.view([-1, 784]), z).mean().item()

    print("The log-likelihood estimate for the  validation set is: ", valid_ll)
    print("The log-likelihood estimate for the test set is: ", test_ll)