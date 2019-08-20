import io
import numpy as np
import pickle

fname = '../data/ca_DATA/wiki.zh.vec'
filename_wordlist = '../data/ca_DATA/wordlist.txt'
filename_emb = '../data/ca_DATA/embinit.save'

fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
data = {}
count = -1
dim = 0
words = []

print ('-> generating wordlist')

for line in fin:
    if count > -1:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        words.append(tokens[0])
        # Get word vectors dimension
        if count == 2:
            dim = len(list(map(float, tokens[1:])))
            print ('dim: ', dim)
        if count % 50000 == 0: # to show the progress when loading the vector file
            print (count)
    count += 1

print ('count of words: ', count)
wordlist = '\n'.join(words) + '\n'
f = open(filename_wordlist,"w+")
f.write(wordlist)
f.close()
print ('-> wordlist saved')

# save emb parameters

print ('-> generating word embedding')

emb = np.empty(shape=[count, dim], dtype=np.float32)
count = -1
newfin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
for line in newfin:
    if count > -1:
        tokens = line.rstrip().split(' ')
        # data[tokens[0]] = list(map(float, tokens[1:]))
        emb[count] = list(map(float, tokens[1:]))
        if count < 5:
            print (count, tokens[0])
        if count % 50000 == 0: # to show the progress when loading the vector file
            print (count)
    count += 1
    
f = open(filename_emb, 'wb')
# cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
pickle.dump(emb, f, protocol=2)
f.close()

print ('-> word embedding saved')
print ('-> DONE.')