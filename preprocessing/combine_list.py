#-*- coding: UTF-8 -*-  
'''
The program to combine wordlist, userlist or prdlist
Used for cross dataset evaluation
'''
import numpy as np
import copy
import random
import math
from functools import cmp_to_key, reduce
import pickle

import sys


class Wordlist(object):
    def __init__(self, filename, maxn = 200000):
        lines = list(map(lambda x: x.split(), open(filename, "r", encoding='utf-8').readlines()[:maxn]))
        self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.voc_reverse = [(item[1], item[0][0]) for item in zip(lines, range(self.size))]
        self.voc = dict(self.voc)
        self.voc_reverse = dict(self.voc_reverse)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1
    
    def getWord(self, id):
        try:
            return self.voc_reverse[id]
        except:
            return -1

isWordList = True # word list need to combine word embedding as well
data_path = '../../data/wordlist'
list1_fname = '%s/wordlist_yelp13.txt' % data_path
list2_fname = '%s/wordlist_IMDB.txt' % data_path
emb1_fname = '%s/embinit_yelp13.save' % data_path
emb2_fname = '%s/embinit_IMDB.save' % data_path
save_lst_fname = '%s/wordlist_IMDB13.txt' % data_path
save_emb_fname = '%s/embinit_IMDB13.save' % data_path

print ('-> loading wordlist')
list1 = Wordlist(list1_fname)
list2 = Wordlist(list2_fname)
print (list1.size)
print (list2.size)

if isWordList:
    print ('-> loading embedding')
    f = open(emb1_fname, 'rb')
    word_vectors_1 = pickle.load(f, encoding='latin1')
    f.close()
    f = open(emb2_fname, 'rb')
    word_vectors_2 = pickle.load(f, encoding='latin1')
    f.close()
    print (np.shape(word_vectors_1))
    print (np.shape(word_vectors_2))
print ('-> loaded')

not_common_words = []

# find all not common words between two datasets
for i in range(list1.size):
    if list2.getID(list1.getWord(i)) == -1:
        # word not in yelp14, should take representation from yelp13 and save to new
        not_common_words.append(list1.getWord(i))

print ('-> not common words')
print (len(not_common_words))
print (not_common_words[0:5])

baseindex = list2.size # based on the list of list2 and append not common items

# write to new wordlist.txt file
f_original = open(list2_fname, "r", encoding='utf-8')
content = f_original.read()
if isWordList: new = word_vectors_2[0:-1]
f = open(save_lst_fname, "a", encoding='utf-8')
f.write(content)
f_original.close()

print ('start merge')
for i, word in enumerate(not_common_words):
    f.write(word+'\n')
    if isWordList:
        new = np.append(new, [word_vectors_1[list1.getID(word)]], axis=0)
    if i % 100 == 0:
        sys.stdout.write(str(i)+' ')
        sys.stdout.flush()

f.close()

if isWordList:
    f = open(save_emb_fname, 'wb')
    pickle.dump(new, f, protocol=2)
    f.close()
# verify
print ('-> combined verification')
if isWordList:
    print ('combined word vectors shape: ', np.shape(new))
tmp = Wordlist(save_lst_fname)
print ('combined list size: ', tmp.size)

print ('-> DONE')