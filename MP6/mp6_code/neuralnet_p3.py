# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        """
        1) DO NOT change the name of self.encoder & self.decoder
        2) Both of them need to be subclass of torch.nn.Module and callable, like
           output = self.encoder(input)
        """
        self.encoder = nn.Sequential(
                                    nn.Conv2d(in_size,out_size,kernel_size=5),
                                    nn.ReLU(True))
        self.decoder = nn.Sequential(
                                    nn.ConvTranspose2d(out_size,in_size,kernel_size=5),
                                    nn.ReLU(True))
        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(),lrate,weight_decay = 1e-5)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return xhat: an (N, out_size) torch tensor of output from the network
        """
        return self.decoder(self.encoder(x))

    def step(self, x):
        # x [100, 784]
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        xhat = self.forward(x)
        print('x: ',x)
        print('xhat: ', xhat)
        L = self.loss_fn(xhat, x)
        print(L)
        L.backward()
        self.optimizer.step()
        return L

def fit(train_set,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, out_size) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return xhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    lrate = 0.01
    losses = np.zeros(n_iter)
    net = NeuralNet(lrate,nn.MSELoss(),1,32)
    print(len(train_set))
    for j in range(10):
        for i in range(n_iter):
            # input to encoder needs to be [batch_size,1,28,28]
            #print(i)
            #print((i * batch_size) % len(train_set), " to ", (i * batch_size) % len(train_set) + batch_size)
            data = train_set[(i * batch_size) % len(train_set):(i * batch_size) % len(train_set) + batch_size]
            #print(data.shape)
            print('data: ', data)
            data = data.view(100,1,28,28)
            losses[i] = net.step(data)
    dev_set = dev_set.view(len(dev_set),1,28,28)
    xhats = dev_set.view(len(dev_set), 784).detach().numpy()
    # should return  [M,784] size
    torch.save(net, 'net_p3.model')
    return losses,xhats, net
