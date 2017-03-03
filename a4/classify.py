"""
classify.py

1. classify the sentiment of each tweet, pick the top 3 positive and top 1 negative, print these tweets.
Using AFINN lexion.
save the top 10 positive result and top 10 negative result to classify_result.csv
"""

from collections import Counter, deque, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pandas as pd
import numpy as np
import re
import os
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen




target = 'taylorswift13'
tweets = pd.read_csv('datasets.csv')


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful','recommend','loves','excited'])

def tokenize(doc):
    """
    Tokenize a string.
    """
    doc = doc.lower()
    results = doc.split()
    for it in range(0, len(results)):
    	results[it] = re.sub('\A\W+', '', results[it])
    	results[it] = re.sub('\W+\Z', '', results[it])
     
    return np.array(results)
    pass

def lexicon_scores2(names,tokens,affinn):
    """
    use affinn file to predict. 
    """
    pos_score = []
    neg_score = []

    for i in range(0, len(names)):
        temp_p = 0
        temp_n = 0
        for word in tokens[i]:
            word = word.lower()
            if word in affinn:
                if affinn[word] > 0:
                    temp_p += affinn[word]
                else:
                    temp_n += (-1) * affinn[word]
        pos_score.append(temp_p)
        neg_score.append(temp_n)
    return pos_score,neg_score

def lexicon_scores(names, tokens):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.
    """
    result = {}
    pos_score = []
    neg_score = []
    for i in range(0, len(names)):
    	"""
    	result[names[i]] = {}
    	result[names[i]]['pos'] = 0
    	result[names[i]]['neg'] = 0
    	"""
    	temp_p = 0
    	temp_n = 0
    	for word in tokens[i]:
    		word = word.lower()
    		if word in neg_words:
    			#result[names[i]]['neg'] += 1
    			temp_n += 1
    		if word in pos_words:
    			temp_p += 1
    			#result[names[i]]['pos'] += 1
    	pos_score.append(temp_p)
    	neg_score.append(temp_n)
    return pos_score,neg_score




names = tweets['name'].tolist()
text = tweets['text'].tolist()

#download the AFINN
url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(BytesIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')
affinn = dict()
for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        affinn[parts[0].decode("utf-8")] = int(parts[1])


tokens = [ tokenize(t) for t in text]

pos_score, neg_score = lexicon_scores(names,tokens)
tk = [tokenize(t) for t in text]
p2,n2 = lexicon_scores2(names,tk,affinn)
pos = sorted(range(len(pos_score)), key=lambda k: pos_score[k], reverse = True)
neg = sorted(range(len(neg_score)), key=lambda k: neg_score[k],reverse = True)

pos2 = sorted(range(len(p2)), key = lambda k: p2[k], reverse = True)
neg2 = sorted(range(len(n2)), key = lambda k: n2[k], reverse = True)

print('top 3 positive tweets is posted by first is %s, second is %s, third is %s:' % (names[pos2[0]], names[pos2[1]], names[pos2[2]]))
print('top positive tweets is:')
print('first is: %s, second is: %s, third is: %s' % (text[pos2[0]], text[pos2[1]], text[pos2[2]]))
print('top negative tweets is posted by %s' % names[neg2[0]])
print('top negative tweets is %s' % text[neg2[0]])

#save top 10 pos and top 10 neg to one .csv file

classify_file = {'pos_name': [names[i] for i in pos2[:10]],
                'pos_content': [text[i] for i in pos2[:10]],
                'neg_name': [names[i] for i in neg2[:10]],
                'neg_content': [text[i] for i in neg2[:10]]}


df_classify = pd.DataFrame(classify_file)
df_classify.to_csv('classify_result.csv')
