# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2018
import copy
import numpy as np
import math
"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

def signum(w,x):
    result = np.dot(w,x)
    if result > 0:
        return 1
    else:
        return 0

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weights = np.zeros(len(train_set[1]))
    weights = np.append(weights,0)
    for count in range(max_iter):
        for i in range(len(train_set)):
            features = np.append(train_set[i],1)
            if signum(weights,features) != train_labels[i]:
                if train_labels[i] == 0:
                    label = -1
                else:
                    label = 1
                weights = np.add(weights, learning_rate*label*features)
    return weights[:-1], weights[-1]

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    w,b = trainPerceptron(train_set,train_labels,learning_rate,max_iter)
    w = np.append(w,b)
    preds = np.array([])
    for features in dev_set:
        features = np.append(features,[1])
        preds = np.append(preds,signum(w,features))
    return preds

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def LR_guess(w,x):
    return sigmoid(np.dot(w,x)) # maybe round this?

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weights = np.zeros(len(train_set[1]))
    weights = np.append(weights, 0)
    for count in range(max_iter):
        deriv = np.zeros(len(train_set[1]) + 1)
        for i in range(len(train_set)):
            features = np.append(train_set[i], 1)
            guess = LR_guess(weights,features)
            deriv = np.add(deriv, features*(guess - train_labels[i]))
        weights = np.add(weights,-learning_rate*(1/len(train_set))*deriv)
    return weights[:-1], weights[-1]

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    w, b = trainLR(train_set, train_labels, learning_rate, max_iter)
    w = np.append(w, b)
    preds = np.array([])
    for features in dev_set:
        features = np.append(features, [1])
        preds = np.append(preds, round(LR_guess(w, features)))
    return preds

def eucDist(x,y):
    #dist = 0
    #for i in range(len(x)):
    #    dist += math.sqrt((x[i] -y[i])**2)
    return math.sqrt((x -y)**2)

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    guesses = []
    for dev_image in dev_set:
        diff_array = [(float('inf'),0)]*k
        #all_diffs = [(float('inf'),0)]*len(train_set)
        dev_sum = np.sum(dev_image)
        for i in range(len(train_set)):
            diff = dev_image - train_set[i]
            finalDist = np.sum(np.absolute(diff))
                #devPixel = [dev_image[j],dev_image[j+1],dev_image[j+2]]
            #    trainPixel = [train_set[i][j],train_set[i][j+1],train_set[i][j+2]]
            #    diff = np.add(diff,eucDist(devPixel,trainPixel))
            #diff = eucDist(train_sums[i],dev_sum)
            #print(finalDist)
            #all_diffs[i] = (finalDist,i)
            if finalDist < diff_array[0][0]:
                l = 0
                #print(finalDist)
                while l < len(diff_array) and finalDist < diff_array[l][0]:
                    l += 1
                prev = (finalDist, i)
                for m in reversed(range(l)):
                    temp = copy.deepcopy(diff_array[m])
                    diff_array[m] = prev
                    prev = temp
                #print(diff_array)
        #np.sort(all_diffs)
        #diff_array = all_diffs[:k]
        #print(diff_array)
        #print('compared')
        noAnCount = 0
        anCount = 0
        for closest in diff_array:
            if train_labels[closest[1]] == 1:
                anCount += 1
            else:
                noAnCount += 1
        #print('anCount',anCount)
        #print('noAnCount',noAnCount)
        if anCount > noAnCount:
            guesses.append(1)
        else:
            guesses.append(0)
        #print('guessed')
    return guesses
