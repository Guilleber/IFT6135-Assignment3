import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt
import samplers
import argparse

class Classifier(nn.Module):
    """
    Class to create a disctrimiator/critic
    """
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.input_size = input_size

        self.layer1 = nn.Linear(self.input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out

def train(metric, dist1, dist2, use_cuda=False,setup=3):
    """
    Function that train the discriminator/ critic by maximizing the JSD/ cross entropy/ WS
    """
    if setup == 4 :
        D = Classifier(1)
    else:
        D = Classifier(2)
    if use_cuda:
        D.cuda()
    optimizer = SGD(D.parameters(), lr=0.001)
    if metric == 'WD':
        dist_u = samplers.distribution2(batch_size=256)
        lambda_ = 10

    for it in range(10000):
        optimizer.zero_grad()
        x_1 = torch.from_numpy(next(dist1)).float()
        x_2 = torch.from_numpy(next(dist2)).float()
        if use_cuda:
            x_1 = x_1.cuda()
            x_2 = x_2.cuda()
        y_1 = D(x_1)
        y_2 = D(x_2)

        if metric == 'JSD':
            y_1 = F.sigmoid(y_1)
            y_2 = F.sigmoid(y_2)
            loss = - torch.mean(torch.log(y_1)) - torch.mean(torch.log(1 - y_2))
        if metric == 'WD':
            a = torch.from_numpy(next(dist_u)).float()
            if use_cuda:
                a = a.cuda()
            x_z = a*x_1 + (1 - a)*x_2
            x_z.requires_grad = True
            y_z = D(x_z)
            grad_z = torch.autograd.grad(y_z, x_z, grad_outputs=torch.ones(y_z.size()).cuda() if use_cuda else torch.ones(y_z.size()), retain_graph=True, create_graph=True, only_inputs=True)[0]
            loss = -(torch.mean(y_1) - torch.mean(y_2) - lambda_*torch.mean((torch.norm(grad_z, p=2, dim=1)-1)**2))
            
        loss.backward()
        optimizer.step()

    return D

def estimate(metric, dist1, dist2, D, use_cuda=False):
    x_1 = torch.from_numpy(next(dist1)).float()
    x_2 = torch.from_numpy(next(dist2)).float()
    if use_cuda:
        x_1 = x_1.cuda()
        x_2 = x_2.cuda()
    y_1 = D(x_1)
    y_2 = D(x_2)

    if metric == 'JSD':
        y_1 = F.sigmoid(y_1)
        y_2 = F.sigmoid(y_2)
        return np.log(2) + (0.5*torch.mean(torch.log(y_1)) + 0.5*torch.mean(torch.log(1 - y_2))).data.cpu().numpy()
    if metric == 'WD':
        return (torch.mean(y_1) - torch.mean(y_2)).data.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='JSD', help="options are 'WS' or 'JSD' or 'ce'")
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--setup', type=int, default=3)
    args = parser.parse_args()


    ### setup for question 3
    if args.setup == 3:
        data_points = ([],[])
        dist1 = samplers.distribution1(0, batch_size=256)
        for x in np.arange(-1.0, 1.01, 0.1):
            x = np.around(x, decimals=1)
            print("x = " + str(x))
            dist2 = samplers.distribution1(x, batch_size=256)
            D = train(args.metric, dist1, dist2, args.use_cuda)
            y = estimate(args.metric, dist1, dist2, D, args.use_cuda)
            data_points[0].append(x)
            data_points[1].append(y)

        plt.plot(data_points[0], data_points[1], '.')
        plt.xlim(-1, 1)
        plt.show()

    ### setup for question 4
    ### using the provided script density_estimation.py
    if args.setup == 4:

        # plot p0 and p1
        plt.figure()

        # empirical
        xx = torch.randn(10000)
        f = lambda x: torch.tanh(x*2+1) + x*0.75
        d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
        plt.hist(f(xx), 100, alpha=0.5, density=1)
        plt.hist(xx, 100, alpha=0.5, density=1)
        plt.xlim(-5,5)
        # exact
        xx = np.linspace(-5,5,1000)
        N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
        plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
        plt.plot(xx, N(xx))


        ############### import the sampler ``samplers.distribution4''
        ############### train a discriminator on distribution4 and standard gaussian
        ############### estimate the density of distribution4

        #######--- INSERT YOUR CODE BELOW ---#######

        dist4 = samplers.distribution4(batch_size=256)
        dist3 = samplers.distribution3(batch_size=256)

        D = train('JSD', dist4, dist3, args.use_cuda, args.setup)

        #dist1 = samplers.distribution4(batch_size=1)
        #dist0 = samplers.distribution3(batch_size=1)
        input = torch.from_numpy(xx).float().cuda() if args.use_cuda else torch.from_numpy(xx).float()
        input = torch.unsqueeze(input, 1)
        D_xx = torch.squeeze(F.sigmoid(D(input)), 1).data.cpu().numpy()
        f_estimate = N(xx)*D_xx/(1-D_xx)



        ############### plotting things
        ############### (1) plot the output of your trained discriminator
        ############### (2) plot the estimated density contrasted with the true density



        r = D_xx # evaluate xx using your discriminator; replace xx with the output
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(xx,r)
        plt.title(r'$D(x)$')

        estimate = f_estimate # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
        plt.subplot(1,2,2)
        plt.plot(xx,estimate)
        plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
        plt.legend(['Estimated','True'])
        plt.title('Estimated vs True')
        plt.show()
