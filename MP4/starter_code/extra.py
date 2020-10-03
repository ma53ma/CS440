from collections import defaultdict
import math
import numpy as np
import queue

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

def extra(train,test):
    '''
    TODO: implement improved viterbi algorithm for extra credits.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
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
    aposSDict = {}
    aposSScale = {}
    aposSProb = {}
    aposSCount = 0
    lyDict = {}
    lyCount = 0
    lyScale = {}
    lyProb = {}
    nAposDict = {}
    nAposScale = {}
    nAposCount = 0
    nAposProb = {}
    numDict = {}
    numScale = {}
    numCount= 0
    numProb = {}
    hyphDict = {}
    hyphScale = {}
    hyphCount = 0
    hyphProb = {}
    edDict = {}
    edScale = {}
    edCount = 0
    edProb = {}
    ingDict = {}
    ingScale = {}
    ingCount = 0
    ingProb = {}
    mDict = {}
    mScale = {}
    mProb = {}
    mCount = 0
    for sent in train:
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
        if word[len(word) - 1] == 'm':
            mCount += 1
            if tag in mDict:
                mDict[tag] = mDict[tag] + 1
            else:
                mDict[tag] = 1
        if word[len(word) - 2] == "'" and word[len(word) - 1] == "s":
            aposSCount += 1
            if tag in aposSDict:
                aposSDict[tag] = aposSDict[tag] + 1
            else:
                aposSDict[tag] = 1
        if word[len(word) - 2] == "n" and word[len(word) - 1] == "'":
            nAposCount += 1
            if tag in nAposDict:
                nAposDict[tag] = nAposDict[tag] + 1
            else:
                nAposDict[tag] = 1
        if word[len(word) - 2] == "l" and word[len(word) - 1] == "y":
            lyCount += 1
            if tag in lyDict:
                lyDict[tag] = lyDict[tag] + 1
            else:
                lyDict[tag] = 1
        if '1' in word or '2' in word or '3' in word or '4' in word or '5' in word or '6' in word or '7' in word or '8' in word or '9' in word:
            numCount += 1
            if tag in numDict:
                numDict[tag] = numDict[tag] + 1
            else:
                numDict[tag] = 1
        if '-' in word:
            hyphCount += 1
            if tag in hyphDict:
                hyphDict[tag] = hyphDict[tag] + 1
            else:
                hyphDict[tag] = 1
        if word[len(word) - 2] == "e" and word[len(word) - 1] == "d":
            edCount += 1
            if tag in edDict:
                edDict[tag] = edDict[tag] + 1
            else:
                edDict[tag] = 1
        if len(word) >= 3 and word[len(word) - 3] == "i" and word[len(word) - 2] == "n" and word[len(word) - 1] == "g":
            ingCount += 1
            if tag in ingDict:
                ingDict[tag] = ingDict[tag] + 1
            else:
                ingDict[tag] = 1
        if tag in hapaxDict:
            hapaxDict[tag] = hapaxDict[tag] + 1
        else:
            hapaxDict[tag] = 1
    for tag1 in uniqueTagDict.keys():
        if tag1 in hapaxDict:
            hapaxScale[tag1] = (hapaxDict[tag1] + alpha) / (len(singleWords) + alpha * len(uniqueTagDict))
        else:
            hapaxScale[tag1] = alpha / (len(singleWords) + alpha * len(uniqueTagDict))
        if tag1 in lyDict:
            lyScale[tag1] = (lyDict[tag1] + alpha) / (lyCount + alpha*(len(uniqueTagDict)))
        else:
            lyScale[tag1] = alpha / (lyCount + alpha*(len(uniqueTagDict)))
        if tag1 in aposSDict:
            aposSScale[tag1] = (aposSDict[tag1] + alpha)/ (aposSCount + alpha*(len(uniqueWordDict) + 1))
        else:
            aposSScale[tag1] = alpha/ (aposSCount + alpha*(len(uniqueWordDict) + 1))
        if tag1 in nAposDict:
            nAposScale[tag1] = (nAposDict[tag1] + alpha)/ (nAposCount + alpha*(len(uniqueWordDict) + 1))
        else:
            nAposScale[tag1] = alpha/ (nAposCount + alpha*(len(uniqueWordDict) + 1))
        if tag1 in numDict:
            numScale[tag1] = (numDict[tag1] + alpha)/ (numCount + alpha*(len(uniqueWordDict) + 1))
        else:
            numScale[tag1] = alpha / (numCount + alpha * (len(uniqueWordDict) + 1))
        if tag1 in hyphDict:
            hyphScale[tag1] = (hyphDict[tag] + alpha) / (hyphCount + alpha*(len(uniqueWordDict) + 1))
        else:
            hyphScale[tag1] = alpha / (hyphCount + alpha*(len(uniqueWordDict) + 1))
        if tag1 in edDict:
            edScale[tag1] = (edDict[tag] + alpha) / (edCount + alpha*(len(uniqueWordDict) + 1))
        else:
            edScale[tag1] = alpha / (edCount + alpha*(len(uniqueWordDict) + 1))
        if tag1 in ingDict:
            ingScale[tag1] = (ingDict[tag] + alpha) / (ingCount + alpha*(len(uniqueWordDict) + 1))
        else:
            ingScale[tag1] = alpha / (ingCount + alpha*(len(uniqueWordDict) + 1))
        if tag1 in mDict:
            mScale[tag1] = (mDict[tag] + alpha) / (mCount + alpha*(len(uniqueWordDict) + 1))
        else:
            mScale[tag1] = alpha / (mCount + alpha * (len(uniqueWordDict) + 1))
        newAlpha = alpha * hapaxScale[tag1]
        lyAlpha = newAlpha * lyScale[tag1]
        aposSAlpha = newAlpha * aposSScale[tag1]
        nAposAlpha = newAlpha * nAposScale[tag1]
        numAlpha = newAlpha * numScale[tag1]
        hyphAlpha = newAlpha * hyphScale[tag1]
        edAlpha = newAlpha * edScale[tag1]
        ingAlpha = newAlpha * ingScale[tag1]
        mAlpha = newAlpha * mScale[tag1]
        tagWordPairNotInDict[tag1] = np.log(newAlpha / (tagCountDict[tag1] + newAlpha * (len(uniqueWordDict) + 1)))
        lyProb[tag1] = np.log(lyAlpha / (tagCountDict[tag1] + newAlpha * (len(uniqueWordDict) + 1)))
        aposSProb[tag1] = np.log(aposSAlpha / (tagCountDict[tag1] + aposSAlpha * (len(uniqueWordDict) + 1)))
        nAposProb[tag1] = np.log(nAposAlpha / (tagCountDict[tag1] + nAposAlpha * (len(uniqueWordDict) + 1)))
        numProb[tag1] = np.log(numAlpha / (tagCountDict[tag1] + numAlpha * (len(uniqueWordDict) + 1)))
        hyphProb[tag1] = np.log(hyphAlpha / (tagCountDict[tag1] + hyphAlpha * (len(uniqueWordDict) + 1)))
        edProb[tag1] = np.log(edAlpha / (tagCountDict[tag1] + edAlpha * (len(uniqueWordDict) + 1)))
        ingProb[tag1] = np.log(ingAlpha / (tagCountDict[tag1] + ingAlpha * (len(uniqueWordDict) + 1)))
        mProb[tag1] = np.log(mAlpha / (tagCountDict[tag1] + mAlpha * (len(uniqueWordDict) + 1)))
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
                transProb[(tag1, tag2)] = np.log(alpha / (tagTagCount[tag1] + alpha * len(uniqueTagTagDict)))
    for sent in test:
        trellis = [[0] * len(sent) for i in range(len(uniqueTagDict))]
        for i in range(len(sent)):
            word = sent[i]
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
                        if '1' in word or '2' in word or '3' in word or '4' in word or '5' in word or '6' in word or '7' in word or '8' in word or '9' in word:
                            emProb = numProb[tag2]
                        elif word[len(word) - 1] == "m":
                            emProb = mProb[tag2]
                        elif len(word) >= 3 and word[len(word) - 3] == "i" and word[len(word) - 2] == "n" and word[
                            len(word) - 1] == "g":
                            emProb = ingProb[tag2]
                        elif word[len(word) - 2] == "l" and word[len(word) - 1] == "y":
                            emProb = lyProb[tag2]
                        elif word[len(word) - 2] == "e" and word[len(word) - 1] == "d":
                            emProb = edProb[tag2]
                        elif word[len(word) - 2] == "n" and word[len(word) - 1] == "'":
                            emProb = nAposProb[tag2]
                        elif word[len(word) - 2] == "'" and word[len(word) - 1] == "s":
                            emProb = aposSProb[tag2]
                        elif '-' in word:
                            emProb = hyphProb[tag2]
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
        finalSent = []
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
        predicts.append(finalSent)
    return predicts