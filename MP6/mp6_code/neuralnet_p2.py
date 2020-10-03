# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss functions
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.hidden = nn.Sequential(nn.Linear(in_size,256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, out_size))
        '''self.fc1 = nn.Linear(in_size,256) #     nn.Linear(256, 128), nn.ReLU(),
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, out_size)'''
        self.optimizer = optim.SGD(self.parameters(),lrate,0.9)


    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        '''x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))'''
        # self.fc4(x)
        return self.hidden(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        L = self.loss_fn(yhat,y)
        L.backward()
        self.optimizer.step()
        return L


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, out_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    lrate = .01
    losses = np.zeros(n_iter)
    yhats = np.zeros(len(dev_set))
    net = NeuralNet(lrate, nn.CrossEntropyLoss(), len(train_set[0]), 5)
    net.optimizer.zero_grad()
    #print("n_iter",n_iter)
    for i in range(10):
        for j in range(n_iter):
            data = train_set[(j * batch_size) % len(train_set):((j + 1) * batch_size)  % len(train_set)]
            data = (data - data.mean()) / data.std()
            labels = train_labels[(j * batch_size) % len(train_set):((j + 1) * batch_size)  % len(train_set)]
            losses[j] = net.step(data, labels)
    print("finished training")
    dev_set = (dev_set - dev_set.mean())/dev_set.std()
    res = net.forward(dev_set)
    for j in range(len(res)):
        yhats[j] = np.argmax(res[j].data)

    torch.save(net, 'net_p2.model')
    return losses, yhats, net
