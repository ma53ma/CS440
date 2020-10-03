"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import queue
import heapq
import numpy as np

tagset = {'NOUN', 'VERB', 'ADJ', 'ADV',
          'PRON', 'DET', 'IN', 'NUM',
          'PART', 'UH', 'X', 'MODAL',
          'CONJ', 'PERIOD', 'PUNCT', 'TO'}

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []
    wordCount = {}
    pairCount = {}
    tagCount = {}
    uniqueTagDict = {}
    for sent in train:
        for pair in sent:
            word = pair[0]
            tag = pair[1]
            if tag not in uniqueTagDict:
                uniqueTagDict[tag] = 0
            if word in wordCount:
                wordCount[word] = wordCount[word] + 1
            else:
                wordCount[word] = 1
            if pair in pairCount:
                pairCount[pair] = pairCount[pair] + 1
            else:
                pairCount[pair] = 1
            if tag in tagCount:
                tagCount[tag] = tagCount[tag] + 1
            else:
                tagCount[tag] = 1
    
    bestTag = ''
    currMax = 0
    for key in tagCount.keys():
        if tagCount[key] > currMax:
            currMax = tagCount[key]
            bestTag = key
    for sent in test:
        predSent = []
        for word in sent:
            if word in wordCount:
                tagQ = []
                for tag in uniqueTagDict.keys():
                    if (word,tag) in pairCount:
                        tagProb = -pairCount[(word,tag)]/wordCount[word]
                    else:
                        tagProb = 0
                    heapq.heappush(tagQ, (tagProb,tag))
                finalPair = heapq.heappop(tagQ)
                predSent.append((word,finalPair[1]))
            else:
                predSent.append((word,bestTag))
        predicts.append(predSent)
    return predicts

class Node:
    prob = 0
    prev = None
    tag = ''
    word = ''

    def __init__(self,prob,prev,tag,word):
        self.prob = prob
        self.prev = prev
        self.tag = tag
        self.word = word

def training(train):
    alpha = .00001
    uniqueTagDict = {}
    tagCountDict = {}
    initDict = {}
    transDict = {}
    numSent = 0
    pairDict = {}
    tagTagCount = {}
    uniqueTagTagDict = {}
    tagVal = 0
    transProb = {}
    tagWordPairInDict = {}
    tagWordPairNotInDict = {}
    initProbDict = {}
    uniqueWordDict = {}
    initTagWordPairInDict = {}
    initTagWordPairNotInDict = {}
    for sent in train:
        numSent+=1
        firstTag = sent[0][1]
        ### Initial occurrences ###
        if firstTag in initDict:
            initDict[firstTag] = initDict[firstTag] + 1
        else:
            initDict[firstTag] = 1
        ###
        for i in range(len(sent)):
            pair1 = sent[i]
            word1 = pair1[0]
            if word1 not in uniqueWordDict:
                uniqueWordDict[word1] = 1
            if i + 1 in range(len(sent)):
                ### Occurrences for first tag leading to another tag ###
                if pair1[1] in tagTagCount:
                    tagTagCount[pair1[1]] = tagTagCount[pair1[1]] + 1
                else:
                    tagTagCount[pair1[1]] = 1
                pair2 = sent[i + 1]
                ### Tag transition occurrences ###
                ### Unique tag transitions ###
                if (pair1[1],pair2[1]) not in uniqueTagTagDict:
                    uniqueTagTagDict[(pair1[1],pair2[1])] = 1
                if (pair1[1],pair2[1]) in transDict:
                    transDict[(pair1[1],pair2[1])] = transDict[(pair1[1],pair2[1])] + 1
                else:
                    transDict[(pair1[1], pair2[1])] = 1
                ###
            ### Counting tag occurrences ###
            word = pair1[0]
            tag = pair1[1]
            if tag in tagCountDict:
                tagCountDict[tag] = tagCountDict[tag] + 1
            else:
                tagCountDict[tag] = 1
            ### Counting unique tag occurrences ###
            if tag not in uniqueTagDict:
                uniqueTagDict[tag] = tagVal
                tagVal+=1
            ### Counting word,tag pair occurrences
            if tag not in pairDict:
                pairDict[tag] = {word : 1}
            else:
                tagTempDict = pairDict[tag]
                if word in tagTempDict:
                    tagTempDict[word] = tagTempDict[word] + 1
                else:
                    tagTempDict[word] = 1
    for tag1 in uniqueTagDict.keys():
        tagWordPairNotInDict[tag1] = np.log(alpha / (tagCountDict[tag1] + alpha*(len(uniqueWordDict) + 1)))
        initTagWordPairNotInDict[tag1] = np.log(alpha / (tagCountDict[tag] + alpha * len(pairDict[tag])))
        if tag1 in initDict:
            initProbDict[tag1] = np.log((initDict[tag1] + alpha / (numSent + len(uniqueTagDict) * alpha)))
        else:
            initProbDict[tag1] = np.log((alpha / (numSent + len(uniqueTagDict) * alpha)))
        for tag2 in uniqueTagDict.keys():
            if (tag1, tag2) in transDict:
                transProb[(tag1,tag2)] = np.log((transDict[(tag1,tag2)] + alpha)/ (tagTagCount[tag1] + alpha*len(uniqueTagTagDict)))
            else:
                transProb[(tag1, tag2)] = np.log(alpha / (tagTagCount[tag1] + alpha * len(uniqueTagTagDict)))
    for tag in pairDict.keys():
        for word in pairDict[tag].keys():
            tagWordPairInDict[(tag,word)] = np.log((pairDict[tag][word] + alpha) / (tagCountDict[tag] + alpha*(len(uniqueWordDict) + 1)))
            initTagWordPairInDict[(tag,word)] = np.log((pairDict[tag][word] + alpha) / (tagCountDict[tag] + alpha * len(pairDict[tag])))

    return uniqueTagDict, tagCountDict, initDict, transDict, pairDict, tagTagCount, uniqueTagTagDict, transProb, tagWordPairInDict, tagWordPairNotInDict, initProbDict, uniqueWordDict, initTagWordPairInDict, initTagWordPairNotInDict

def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    #uniqueTagDict, tagCountDict, initDict, transDict, pairDict, tagTagCount, uniqueTagTagDict, transProb, tagWordPairInDict, tagWordPairNotInDict, initProbDict, uniqueWordDict, initTagWordPairInDict, initTagWordPairNotInDict = training(train)
    predicts = []
    for sent in test:
        #trellis = [[0]*len(sent) for i in range(len(uniqueTagDict))]
        finalSent = []
        for i in range(len(sent)):
            word = sent[i]
            finalSent.append((word,'NOUN'))
            '''
            if i == 0:
                for tag in uniqueTagDict.keys():
                    tagNum = uniqueTagDict[tag]
                    initTagProb = initProbDict[tag]
                    if word in pairDict[tag]:
                        pairProb = initTagWordPairInDict[(tag,word)]
                    else:
                        pairProb = initTagWordPairNotInDict[tag]
                    finalTagProb = initTagProb + pairProb
                    trellisNode = Node(finalTagProb,None,tag,word)
                    trellis[tagNum][i] = trellisNode
            else:
                for tag2 in uniqueTagDict.keys():
                    ## EMISSION PROBABILITIES ##
                    if word in pairDict[tag2]:
                        emProb = tagWordPairInDict[(tag2, word)]
                    else:
                        emProb = tagWordPairNotInDict[tag2]
                    ####
                    bestProb = float('-inf')
                    bestTag = ''
                    for tag1 in uniqueTagDict.keys():
                        finalTagProb = transProb[(tag1, tag2)] + emProb + trellis[uniqueTagDict[tag1]][i - 1].prob
                        if finalTagProb > bestProb:
                            bestProb = finalTagProb
                            bestTag = tag1
                    trellis[uniqueTagDict[tag2]][i] = Node(bestProb, trellis[uniqueTagDict[bestTag]][i - 1], tag2, word)
        best = queue.PriorityQueue()
        for tag in uniqueTagDict.keys():
            tagIdx = uniqueTagDict[tag]
            curr = trellis[tagIdx][len(sent) - 1]
            best.put((-curr.prob,curr))
        bestLast = best.get()
        bestLastNode = bestLast[1]
        finalSent.append((bestLastNode.word,bestLastNode.tag))
        prev = bestLastNode.prev
        while prev is not None:
            finalSent.append((prev.word,prev.tag))
            prev = prev.prev
        finalSent.reverse()
        '''
        predicts.append(finalSent)
    return predicts

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    alpha = .00001
    uniqueTagDict = {}
    tagCountDict = {}
    initDict = {}
    transDict = {}
    numSent = 0
    pairDict = {}
    tagTagCount = {}
    uniqueTagTagDict = {}
    tagVal = 0
    predicts = []
    transProb = {}
    tagWordPairInDict = {}
    tagWordPairNotInDict = {}
    initProbDict = {}
    uniqueWordDict = {}
    initTagWordPairInDict = {}
    initTagWordPairNotInDict = {}
    singleWords = {}
    hapaxDict = {}
    hapaxScale = {}
    '''    for sent in train:
        numSent += 1
        firstTag = sent[0][1]
        ### Initial occurrences ###
        if firstTag in initDict:
            initDict[firstTag] = initDict[firstTag] + 1
        else:
            initDict[firstTag] = 1
        ###
        for i in range(len(sent)):
            pair1 = sent[i]
            word1 = pair1[0]
            if word1 not in uniqueWordDict:
                uniqueWordDict[word1] = 1
                singleWords[word1] = pair1[1]
            else:
                uniqueWordDict[word1] = uniqueWordDict[word1] + 1
                if word1 in singleWords:
                    del singleWords[word1]
            if i + 1 in range(len(sent)):
                ### Occurrences for first tag leading to another tag ###
                if pair1[1] in tagTagCount:
                    tagTagCount[pair1[1]] = tagTagCount[pair1[1]] + 1
                else:
                    tagTagCount[pair1[1]] = 1
                pair2 = sent[i + 1]
                ### Tag transition occurrences ###
                ### Unique tag transitions ###
                if (pair1[1], pair2[1]) not in uniqueTagTagDict:
                    uniqueTagTagDict[(pair1[1], pair2[1])] = 1
                if (pair1[1], pair2[1]) in transDict:
                    transDict[(pair1[1], pair2[1])] = transDict[(pair1[1], pair2[1])] + 1
                else:
                    transDict[(pair1[1], pair2[1])] = 1
                ###
            ### Counting tag occurrences ###
            word = pair1[0]
            tag = pair1[1]
            if tag in tagCountDict:
                tagCountDict[tag] = tagCountDict[tag] + 1
            else:
                tagCountDict[tag] = 1
            ### Counting unique tag occurrences ###
            if tag not in uniqueTagDict:
                uniqueTagDict[tag] = tagVal
                tagVal += 1
            ### Counting word,tag pair occurrences
            if tag not in pairDict:
                pairDict[tag] = {word: 1}
            else:
                tagTempDict = pairDict[tag]
                if word in tagTempDict:
                    tagTempDict[word] = tagTempDict[word] + 1
                else:
                    tagTempDict[word] = 1
    for tag in pairDict.keys():
        for word in pairDict[tag].keys():
            tagWordPairInDict[(tag, word)] = np.log(
                (pairDict[tag][word] + alpha) / (tagCountDict[tag] + alpha * (len(uniqueWordDict) + 1)))
            initTagWordPairInDict[(tag, word)] = np.log(
                (pairDict[tag][word] + alpha) / (tagCountDict[tag] + alpha * len(pairDict[tag])))
    ### HAPAX WORDS ###
    for word in singleWords.keys():
        tag = singleWords[word]
        if tag in hapaxDict:
            hapaxDict[tag] = hapaxDict[tag] + 1
        else:
            hapaxDict[tag] = 1
    for tag1 in uniqueTagDict.keys():
        if tag1 in hapaxDict:
            hapaxScale[tag1] = (hapaxDict[tag1] + alpha) / (len(singleWords) + alpha*len(uniqueTagDict))
        else:
            hapaxScale[tag1] = alpha / (len(singleWords) + alpha*len(uniqueTagDict))
        newAlpha = alpha * hapaxScale[tag1]
        tagWordPairNotInDict[tag1] = np.log(newAlpha / (tagCountDict[tag1] + newAlpha*(len(uniqueWordDict) + 1)))
        initTagWordPairNotInDict[tag1] = np.log(alpha / (tagCountDict[tag1] + alpha * len(pairDict[tag1])))
        if tag1 in initDict:
            initProbDict[tag1] = np.log(((initDict[tag1] + alpha) / (numSent + len(uniqueTagDict) * alpha)))
        else:
            initProbDict[tag1] = np.log((alpha / (numSent + len(uniqueTagDict) * alpha)))
        for tag2 in uniqueTagDict.keys():
            if (tag1, tag2) in transDict:
                transProb[(tag1, tag2)] = np.log(
                    (transDict[(tag1, tag2)] + alpha) / (tagTagCount[tag1] + alpha * len(uniqueTagTagDict)))
            else:
                transProb[(tag1, tag2)] = np.log(alpha / (tagTagCount[tag1] + alpha * len(uniqueTagTagDict)))'''

    print("finished training")
    for sent in test:
        finalSent = []
        #trellis = [[0] * len(sent) for i in range(len(uniqueTagDict))]
        for i in range(len(sent)):
            word = sent[i]
            finalSent.append((word,'NOUN'))
            '''
            if i == 0:
                for tag in uniqueTagDict.keys():
                    tagNum = uniqueTagDict[tag]
                    initTagProb = initProbDict[tag]
                    if word in pairDict[tag]:
                        pairProb = initTagWordPairInDict[(tag, word)]
                    else:
                        pairProb = initTagWordPairNotInDict[tag]
                    finalTagProb = initTagProb + pairProb
                    trellisNode = Node(finalTagProb, None, tag, word)
                    trellis[tagNum][i] = trellisNode
            else:
                for tag2 in uniqueTagDict.keys():
                    ## EMISSION PROBABILITIES ##
                    if word in pairDict[tag2]:
                        emProb = tagWordPairInDict[(tag2, word)]
                        #print(emProb,'seen')
                    else:
                        emProb = tagWordPairNotInDict[tag2]
                        #print(emProb,'unseen')
                    ####
                    bestProb = float('-inf')
                    bestTag = ''
                    for tag1 in uniqueTagDict.keys():
                        finalTagProb = transProb[(tag1, tag2)] + emProb + trellis[uniqueTagDict[tag1]][i - 1].prob
                        if finalTagProb > bestProb:
                            bestProb = finalTagProb
                            bestTag = tag1
                    trellis[uniqueTagDict[tag2]][i] = Node(bestProb, trellis[uniqueTagDict[bestTag]][i - 1], tag2, word)
        best = queue.PriorityQueue()
        for tag in uniqueTagDict.keys():
            tagIdx = uniqueTagDict[tag]
            curr = trellis[tagIdx][len(sent) - 1]
            best.put((-curr.prob, curr))
        bestLast = best.get()
        bestLastNode = bestLast[1]
        finalSent.append((bestLastNode.word, bestLastNode.tag))
        prev = bestLastNode.prev
        while prev is not None:
            finalSent.append((prev.word, prev.tag))
            prev = prev.prev
        finalSent.reverse()
        '''
        predicts.append(finalSent)
    return predicts