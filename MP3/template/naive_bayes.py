# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # for each set in the training set
    # sort training set and see how many of given word are in there
    # iterate through it and create dict of word and frequency and +/-
    pos_dict = {}
    neg_dict = {}
    word_total = 0
    final_labels = []
    pos_words = 0
    neg_words = 0
    unique_pos_words = {}
    unique_neg_words = {}
    for i in range(len(train_set)):
        for word in train_set[i]:
            if train_labels[i]:
                pos_words = pos_words + 1
                if word not in unique_pos_words:
                    unique_pos_words[word] = 1
                if word in pos_dict:
                    pos_dict[word] = pos_dict[word] + 1
                else:
                    pos_dict[word] = 1
            else:
                neg_words = neg_words + 1
                if word not in unique_neg_words:
                    unique_neg_words[word] = 1
                if word in neg_dict:
                    neg_dict[word] = neg_dict[word] + 1
                else:
                    neg_dict[word] = 1
    # now we have probability of a word given positive or negative, and probability of positive or negative
    # now need to calculate posterior of each document
    for i in range(len(dev_set)):
        pos_post = numpy.log(pos_prior)
        neg_post = numpy.log(1 - pos_prior)
        pos_list = []
        neg_list = []
        for word in dev_set[i]:
            # positive posterior
            if word in pos_dict:
                pos_like = numpy.log((pos_dict[word] + smoothing_parameter)/(pos_words + len(unique_pos_words)*smoothing_parameter))
            else:
                pos_like = numpy.log(smoothing_parameter / (pos_words + len(unique_pos_words)*smoothing_parameter))
            pos_list.append(pos_like)
            if word in neg_dict:
                neg_like = numpy.log((neg_dict[word] + smoothing_parameter)/(neg_words + len(unique_neg_words)*smoothing_parameter))
            else:
                neg_like = numpy.log(smoothing_parameter/(neg_words + len(unique_neg_words)*smoothing_parameter))
            neg_list.append(neg_like)
        for word_p in pos_list:
            pos_post = pos_post+word_p
        for word_p in neg_list:
            neg_post = neg_post+word_p
        if pos_post > neg_post:
            final_labels.append(1)
        else:
            final_labels.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return final_labels