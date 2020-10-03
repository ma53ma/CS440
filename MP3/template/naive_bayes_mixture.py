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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
from collections import Counter

def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    pos_uni_dict = {}
    neg_uni_dict = {}
    pos_bi_dict = {}
    neg_bi_dict = {}
    pos_uni_words = 0
    neg_uni_words = 0
    pos_bi_words = 0
    neg_bi_words = 0
    unique_uni_pos_words  = {}
    unique_uni_neg_words = {}
    unique_bi_pos_words = {}
    unique_bi_neg_words = {}
    final_list = []
    #### TRAINING SET ####
    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            word1 = train_set[i][j]
            if train_labels[i]:
                pos_uni_words = pos_uni_words + 1
                if word1 not in unique_uni_pos_words:
                    unique_uni_pos_words[word1] = 1
                #### UNIGRAM ####
                if word1 in pos_uni_dict:
                    pos_uni_dict[word1] = pos_uni_dict[word1] + 1
                else:
                    pos_uni_dict[word1] = 1
                ### BIGRAM ####
                if j + 1 in range(len(train_set[i])):
                    pos_bi_words = pos_bi_words + 1
                    word2 = train_set[i][j+1]
                    if (word1,word2) not in unique_bi_pos_words:
                        unique_bi_pos_words[(word1,word2)] = 1
                    if (word1,word2) in pos_bi_dict:
                        pos_bi_dict[(word1,word2)] = pos_bi_dict[(word1,word2)] + 1
                    else:
                        pos_bi_dict[(word1, word2)] = 1
            else:
                neg_uni_words = neg_uni_words + 1
                if word1 not in unique_uni_neg_words:
                    unique_uni_neg_words[word1] = 1
                #### UNIGRAM ####
                if word1 in neg_uni_dict:
                    neg_uni_dict[word1] = neg_uni_dict[word1] + 1
                else:
                    neg_uni_dict[word1] = 1
                ### BIGRAM ####
                if j + 1 in range(len(train_set[i])):
                    neg_bi_words = neg_bi_words + 1
                    word2 = train_set[i][j + 1]
                    if (word1,word2) not in unique_bi_neg_words:
                        unique_bi_neg_words[(word1,word2)] = 1
                    if (word1, word2) in neg_bi_dict:
                        neg_bi_dict[(word1, word2)] = neg_bi_dict[(word1, word2)] + 1
                    else:
                         neg_bi_dict[(word1, word2)] = 1
    #### DEVELOPMENT SET ####
    for i in range(len(dev_set)):
        pos_uni_post = numpy.log(pos_prior)
        neg_uni_post = numpy.log(1 - pos_prior)
        pos_bi_post = numpy.log(pos_prior)
        neg_bi_post = numpy.log(1 - pos_prior)
        pos_uni_list = []
        neg_uni_list = []
        pos_bi_list = []
        neg_bi_list = []
        for j in range(len(dev_set[i])):
            word1 = dev_set[i][j]
            #### UNIGRAM POS ####
            if word1 in pos_uni_dict:
                pos_uni_like = numpy.log((pos_uni_dict[word1] + unigram_smoothing_parameter) / (pos_uni_words + len(unique_uni_pos_words) * unigram_smoothing_parameter))
            else:
                pos_uni_like = numpy.log(unigram_smoothing_parameter / (pos_uni_words + len(unique_uni_pos_words) * unigram_smoothing_parameter))
            pos_uni_list.append(pos_uni_like)
            #### BIGRAM POS ####
            if j + 1 in range(len(dev_set[i])):
                word2 = dev_set[i][j + 1]
                if (word1,word2) in pos_bi_dict:
                    pos_bi_like = numpy.log((pos_bi_dict[(word1, word2)] + bigram_smoothing_parameter) / (pos_bi_words +  len(unique_bi_pos_words)*bigram_smoothing_parameter)) # len(pos_bi_dict)
                else:
                    pos_bi_like = numpy.log(bigram_smoothing_parameter / (pos_bi_words + len(unique_bi_pos_words)*bigram_smoothing_parameter))
                pos_bi_list.append(pos_bi_like)
            #### UNIGRAM NEG ####
            if word1 in neg_uni_dict:
                neg_uni_like = numpy.log((neg_uni_dict[word1] + unigram_smoothing_parameter) / (neg_uni_words + len(unique_uni_neg_words)*unigram_smoothing_parameter))
            else:
                neg_uni_like = numpy.log(unigram_smoothing_parameter / (neg_uni_words + len(unique_uni_neg_words)*unigram_smoothing_parameter))
            neg_uni_list.append(neg_uni_like)
            #### BIGRAM NEG ####
            if j + 1 in range(len(dev_set[i])):
                word2 = dev_set[i][j + 1]
                if (word1,word2) in neg_bi_dict:
                    neg_bi_like = numpy.log((neg_bi_dict[(word1, word2)] + bigram_smoothing_parameter) / (neg_bi_words + len(unique_bi_neg_words)*bigram_smoothing_parameter))
                else:
                    neg_bi_like = numpy.log(bigram_smoothing_parameter / (neg_bi_words + len(unique_bi_neg_words)*bigram_smoothing_parameter))
                neg_bi_list.append(neg_bi_like)

        for word_p in pos_uni_list:
            pos_uni_post = pos_uni_post + word_p
        for word_p in neg_uni_list:
            neg_uni_post = neg_uni_post + word_p
        for pair_p in pos_bi_list:
            pos_bi_post = pos_bi_post + pair_p
        for pair_p in neg_bi_list:
            neg_bi_post = neg_bi_post + pair_p

        pos_total = (1 - bigram_lambda)*pos_uni_post + bigram_lambda*pos_bi_post
        neg_total = (1- bigram_lambda)*neg_uni_post + bigram_lambda*neg_bi_post
        if pos_total > neg_total:
            final_list.append(1)
        else:
            final_list.append(0)

    return final_list