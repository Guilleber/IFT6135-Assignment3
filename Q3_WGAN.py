import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch import optim
from torch import autograd
from torchvision import transforms
from classify_svhn import get_data_loader
from q3_vae import View
import numpy as np
import matplotlib.pyplot as plt
import samplers
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-t", action="store_true", help="Flag to specify if we train the model")
parser.add_argument("--save_path", type=str, default="q3_gan.pt")
parser.add_argument("--load_path", type=str, default="q3_gan.pt")
parser.add_argument("--batch_size", type=int, default=128, help="Size of the mini-batches")
parser.add_argument("--dimz", type=int, default=100, help="Dimension of the latent variables")
parser.add_argument("--data_dir", type=str, default="svhn.mat", help="SVHN dataset location")
parser.add_argument("--nb_epochs", type=int, default=20, help = "The number of epochs for training")
parser.add_argument("--lam",  type=int, default=10, help="Lambda coefficient for the regularizer in"
                                                            "in the WGAN-GP loss")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for the optimzer")
parser.add_argument("--update_ratio", type=int, default=5, help="The number of updates to  the discriminator"
                                                                 "before one update to the generator")
parser.add_argument("--eps", type=float, default=1e-1, help="Perturbation value to the latent when evaluating")
parser.add_argument("--sample_dir", type=str, default="samples", help="Directory containing samples for"
                                                                      "evaluation")

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
            nn.Conv2d(3, 128, 5, padding=2, stride=2),
            nn.LeakyReLU(0.2),

            # layer2
            nn.Conv2d(128, 256, 5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # layer3
            nn.Conv2d(256, 512, 5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # layer 4
            View(-1, 4*4*512),
            nn.Linear(4 * 4 * 512, 1),
            nn.BatchNorm1d(1),  # may have to remove
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.convs(x)[:, 0]
        return out


class G(nn.Module):

    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz

        self.deconvs = nn.Sequential(
            # layer 1
            nn.Linear(self.dimz, 4 * 4 * 512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(),
            View(-1, 512, 4, 4),

            # layer2
            nn.ConvTranspose2d(512, 256, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # layer 3
            nn.ConvTranspose2d(256, 128, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # layer 4
            nn.ConvTranspose2d(128, 3, 5, padding=2, stride=2, output_padding=1),
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
    return fake - real + lam * (torch.norm(grad, dim=1) - 1)**2


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
    g_optim = optim.Adam(g.parameters(), lr=args.lr, betas=(0, 0.9))
    d_optim = optim.Adam(d.parameters(), lr=args.lr, betas=(0, 0.9))

    # print PIL image
    display = transforms.ToPILImage()

    for epoch in range(args.nb_epochs):
        for i, (batch, label) in enumerate(train):
            # put batch on device
            batch = batch.to(args.device)

            # obtain the discriminator output on real data
            real_prob = d(batch)

            # obtain the discriminator output on the fake data
            z = torch.randn(batch.size()[0], g.dimz, device=args.device)
            fake = g(z)
            fake_prob = d(fake)

            # obtain the gradient term of the WGAN-GP loss
            a = torch.rand(batch.size()[0], 1, 1, 1, device=args.device)
            conv = a * batch + (1. - a) * fake
            d_conv = d(conv)
            grad = autograd.grad(d_conv, conv, torch.ones_like(d_conv).to(args.device),
                                 retain_graph=True, create_graph=True, only_inputs=True)[0]

            # compute the WGAN-GP loss
            loss = wgan_gp_loss(real_prob, fake_prob, grad.view(-1, 3 * 32 * 32), args.lam).mean()

            # minimize the loss
            autograd.backward([loss])

            # update the parameters
            d_optim.step()
            d_optim.zero_grad()

            if i % args.update_ratio == 0. and i > 0:
                # update the generator
                z = torch.randn(batch.size()[0], g.dimz, device=args.device)
                fake = g(z)
                fake_prob = d(fake)
                loss = - fake_prob.mean()
                loss.backward()
                g_optim.step()
                g_optim.zero_grad()

        with torch.no_grad():
            # After training for an epoch, output validation loss
            valid_loss = torch.zeros(1)
            nb_batches = 0
            for i, (batch, label) in enumerate(valid):
                nb_batches += 1
                batch = batch.to(args.device)
                real_prob = d(batch)
                z = torch.randn(batch.size()[0], g.dimz, device=args.device)
                fake = g(z)
                display(((fake[0] + 1.) * 255.).to(device='cpu', copy=True)).show()
                fake_prob = d(fake)
                a = torch.rand(batch.size()[0], 1, 1, 1, device=args.device)

                with torch.enable_grad():
                    conv = a * batch + (1. - a) * fake
                    conv.requires_grad = True
                    d_conv = d(conv)
                    grad = autograd.grad(d_conv, conv, torch.ones_like(d_conv).to(args.device),
                                         retain_graph=False, create_graph=True, only_inputs=True)[0]
                batch_loss = wgan_gp_loss(real_prob, fake_prob, grad.view(-1, 3 * 32 * 32), args.lam)
                valid_loss += batch_loss.mean()
            valid_loss /= nb_batches
            print("After epoch {} the validation loss is: ".format(epoch + 1), valid_loss.item())

    # save the model to be used later
    torch.save(g.state_dict(), save_path)


def evaluation(model):
    """
    Function that generates samples for the qualitative evaluation of the model
    :param model: The model from which we pull samples
    :return:
    """
    with torch.no_grad():
        transf = transforms.ToPILImage()
        z = torch.randn(model.batch_size, model.dimz, device=args.device)
        samples = model.deconvs(z)

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
        samples = model.deconvs(z_)

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
            gz_a = model.deconvs(z_a)[0]
            im = transf(gz_a.to(device='cpu'))
            im.save(os.path.join(int_dir, "i1_img_{}.jpeg".format(a)))

        # interpolate the result of the two g(z) values
        g_z = model.deconvs(z[0:2])
        for a in range(11):
            x_a = a / 10. * g_z[0] + (1. - a / 10.) * g_z[1]
            im = transf(x_a.to(device='cpu'))
            im.save(os.path.join(int_dir, "i2_img_{}.jpeg".format(a)))


if __name__ == "__main__":
    # check for cuda
    device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')
    args.device = device

    # load the dataset
    train, valid, test = get_data_loader(args.data_dir, args.batch_size)

    # Create model. Load or train depending on choice
    g = G(args.batch_size, args.dimz).to(args.device)
    d = D(args.batch_size, args.dimz).to(args.device)
    if args.t:
        train_model(g, d, train, valid, args.save_path)
    else:
        g.load_state_dict(torch.load(args.load_path))
        g.eval()

        evaluation(g)
