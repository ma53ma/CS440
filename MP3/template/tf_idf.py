# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time



def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """


    #tf-idf term: (# of times word w appears in dev doc A/# of words in doc. A)*log(total # docs in train set/(1+# docs in train set w/ word w))
    # return array of words with highest term for each dev. doc
    # TODO: Write your code here
    train_docs = 0
    train_dict = {}
    checked_doc = {}
    #### TRAINING SET ####
    for doc in train_set:
        train_docs += 1
        for word in doc:
            if word not in checked_doc:
                checked_doc[word] = True
                if word in train_dict:
                    train_dict[word] = train_dict[word] + 1
                else:
                    train_dict[word] = 1
        checked_doc = {}
    #### DEVELOPMENT SET ####
    doc_words = 0
    dev_dict = {}
    tf_idf_total_list = []
    tf_idf_doc_dict = {}
    for doc in dev_set:
        for word in doc:
            doc_words += 1
            if word in dev_dict:
                dev_dict[word]+=1
            else:
                dev_dict[word] = 1
        for word in dev_dict.keys():
            if word in train_dict:
                tf_idf_doc_dict[word] = (dev_dict[word]/doc_words)*np.log(train_docs/(1+train_dict[word]))
            else:
                tf_idf_doc_dict[word] = (dev_dict[word] / doc_words) * np.log(train_docs)
        doc_words = 0
        dev_dict = {}
        cur_max = 0
        cur_word = ''
        for word in tf_idf_doc_dict.keys():
            if tf_idf_doc_dict[word] > cur_max:
                cur_word = word
                cur_max = tf_idf_doc_dict[word]
        tf_idf_doc_dict = {}
        tf_idf_total_list.append(cur_word)
    return tf_idf_total_list