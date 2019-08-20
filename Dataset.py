#-*- coding: UTF-8 -*-  
import numpy
import copy
import random
import math
from functools import cmp_to_key, reduce

def genBatch(data):
    m = 0
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence)>m:
                m = len(sentence)
        for i in range(maxsentencenum - len(doc)):
            doc.append([0])
    tmp = list(map(lambda doc: numpy.asarray(list(map(lambda sentence : sentence + [0]*(m - len(sentence)), doc)), dtype = numpy.int32).T, data))
    tmp = reduce(lambda doc,docs : numpy.concatenate((doc,docs),axis = 1),tmp)
    return tmp

def genBatchFloat(data):
    m = 0 
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence) > m:
                m = len(sentence)
        for i in xrange(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence + [-1] * (m - len(sentence)), doc), dtype=numpy.float32).T, data)  # [-1]是加在最前面
    tmp = reduce(lambda doc, docs : numpy.concatenate((doc, docs), axis=1), tmp)

    finalres = []
    for indwords in xrange(0, len(tmp)):
        finalres.append([])
        for indword in xrange(0, len(tmp[indwords])):
            finalres[indwords].append([])
            finalres[indwords][indword] = tmp[indwords][indword] * numpy.ones(200) 
    return numpy.asarray(finalres) 

def genBatchFloatSent(data):
    maxsentencenum = len(data[0])
    for doc in data:
        for i in xrange(maxsentencenum - len(doc)):
            doc.append(0.0)
    try:
        tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence, doc), dtype=numpy.float32), data)  # [-1]是加在最前面
    except:
        pass

    finalres = []
    for indwords in xrange(0, len(tmp)):
        finalres.append([])
        for indword in xrange(0, len(tmp[indwords])):
            finalres[indwords].append([])
            finalres[indwords][indword] = tmp[indwords][indword] * numpy.ones(200) 
    return numpy.asarray(finalres) 
            
def genLenBatch(lengths,maxsentencenum):
    lengths = map(lambda length : numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4),lengths)
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = list(map(lambda x : list(map(lambda y : [1.0 ,0.0][y == -1],x)), mask))
    mask = numpy.asarray(mask,dtype=numpy.float32)
    return mask

def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray(list(map(lambda num : [1.0]*num + [0.0]*(maxnum - num),sentencenum)), dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb, usrdict, prddict, maxbatch, ten2five_classes = False, chinese = False, maxword = 500):
        lines = list(map(lambda x: x.split('\t\t'), open(filename, "r", encoding='utf-8').readlines()))         
        label = numpy.asarray(
            list(map(lambda x: int(x[2])-1, lines)),
            dtype = numpy.int32
        )
        if ten2five_classes:
            label_5_classes = []
            for lab in label:
                if lab % 2 != 0:
                    label_5_classes.append(math.floor(lab / 2))
                else:
                    label_5_classes.append(int(lab / 2))
            print (label[0:10])
            label = numpy.asarray(label_5_classes, dtype = numpy.int32)
            print (label[0:10])

        usr = list(map(lambda line: usrdict.getID(line[0]),lines))
        prd = list(map(lambda line: prddict.getID(line[1]),lines))
        docs = list(map(lambda x: x[3][0:len(x[3])-1], lines))
        docs = list(map(lambda x: x.split('<sssss>'), docs))
        # divide sentence into words
        if chinese:
            import jieba
            # parse sentence into words
            print('-> Chinese parsing words...')
            docs = list(map(lambda doc: list(map(lambda sentence: list(jieba.cut(sentence, cut_all=False)),doc)),docs))
            print('-> Chinese parsing words done.')
        else:
            docs = list(map(lambda doc: list(map(lambda sentence: sentence.split(' '),doc)),docs))
        docs = list(map(lambda doc: list(map(lambda sentence: list(filter(lambda wordid: wordid !=-1,list(map(lambda word: emb.getID(word),sentence)))),doc)),docs))
        print('document number: ', numpy.shape(docs))
        self.num_doc = len(docs)

        tmp = list(zip(docs, label, usr, prd))
        #random.shuffle(tmp)
        tmp = sorted(tmp, key=cmp_to_key(lambda x, y: len(y[0]) - len(x[0])))
        # docs, label, usr, prd = zip(*tmp)
        self.tmp_docs, self.tmp_label, self.tmp_usr, self.tmp_prd = zip(*tmp)
        self.tmp_sentencenum = map(lambda x : len(x), self.tmp_docs)

        # load user memory: which documents(index) posted by user usrid
        self.num_usr = len(self.tmp_usr)
        usr_1 = numpy.array(self.tmp_usr)
        self.usr_mem = []
        for usrid in range(0, len(usrdict.usrlist)):
            self.usr_mem.append(numpy.where(usr_1 == usrid)[0])
            
        self.num_prd = len(self.tmp_prd)
        # load product memory
        prd_1 = numpy.array(self.tmp_prd)
        self.prd_mem = []
        for prdid in range(0, len(prddict.prdlist)):
            self.prd_mem.append(numpy.where(prd_1 == prdid)[0])

        self.tmp_sentencenum = list(map(lambda x : len(x), self.tmp_docs))
        self.epoch = math.ceil(len(docs) / maxbatch)

        self.docs = []
        self.label = []
        self.usr = []
        self.prd = []
        self.wordmask = []
        self.sentencemask = []
        self.maxsentencenum = []
        
        for i in range(self.epoch):
            self.maxsentencenum.append(self.tmp_sentencenum[i * maxbatch])
            docsbatch = genBatch(self.tmp_docs[i * maxbatch:(i + 1) * maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(self.tmp_label[i * maxbatch:(i + 1) * maxbatch], dtype=numpy.int32))
            self.usr.append(numpy.asarray(self.tmp_usr[i * maxbatch:(i + 1) * maxbatch], dtype=numpy.int32))
            self.prd.append(numpy.asarray(self.tmp_prd[i * maxbatch:(i + 1) * maxbatch], dtype=numpy.int32))
            self.wordmask.append(genwordmask(docsbatch))
            self.sentencemask.append(gensentencemask(self.tmp_sentencenum[i * maxbatch:(i + 1) * maxbatch]))

    def gene_usr_context(self, thislist, max_mem):
        # list is the list of usr id in integer
        data = numpy.asarray(list(map(lambda id: self.usr_mem[id], thislist)))
        # Get lengths of each row of data
        lens = numpy.array([len(i) for i in data])
        # Mask of valid places in each row
        mask = numpy.arange(lens.max()) < lens[:,None]
        # Setup output array and put elements from data into masked positions
        out = numpy.empty(shape=mask.shape, dtype=numpy.int32)
        out.fill(0)
        out[mask] = numpy.concatenate(data)
        if lens.max() > max_mem:
            # only take related docs in memory size
            out = out[:,:max_mem]
        else:
            a = numpy.empty(shape=[len(thislist),(max_mem - lens.max())], dtype=numpy.int32)
            a.fill(0)
            out = numpy.append(out, a, axis=1)
        return numpy.asarray(out)

    def gene_prd_context(self, thislist, max_mem):
        # list is the list of prd id in integer
        data = numpy.asarray(list(map(lambda id: self.prd_mem[id], thislist)))
        # Get lengths of each row of data
        lens = numpy.array([len(i) for i in data])
        # Mask of valid places in each row
        mask = numpy.arange(lens.max()) < lens[:,None]
        # Setup output array and put elements from data into masked positions
        out = numpy.empty(shape=mask.shape, dtype=numpy.int32)
        out.fill(0)
        out[mask] = numpy.concatenate(data)
        if lens.max() > max_mem:
            # only take related docs in memory size
            out = out[:,:max_mem]
        else:
            a = numpy.empty(shape=[len(thislist),(max_mem - lens.max())], dtype=numpy.int32)
            a.fill(0)
            out = numpy.append(out, a, axis=1)
        # print numpy.shape(out)
        return numpy.asarray(out)
        

class Wordlist(object):
    def __init__(self, filename, maxn = 500000):
        lines = list(map(lambda x: x.split(), open(filename, "r", encoding='utf-8').readlines()[:maxn]))
        self.size = len(lines)
        print ('wordlist size: ', self.size)

        self.voc = [(self.getInside(item[0]), item[1]) for item in zip(lines, range(self.size))]
        self.voc_reverse = [(item[1], self.getInside(item[0])) for item in zip(lines, range(self.size))]
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
    
    def getInside(self, item):
        try:
            return item[0]
        except:
            print ('-- Warning: no word inside, this item: ', item)
            return '    '

class Usrlist(object):
    def __init__(self, filename, maxn=100000):
        lines = list(map(lambda x: x.split(), open(filename, "r", encoding='utf-8').readlines()[:maxn]))
        self.size = len(lines)

        self.usrlist = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.usrdict = dict(self.usrlist)
        self.usrlist_reverse = [(item[1], item[0][0]) for item in zip(lines, range(self.size))]
        self.usrdict_reverse = dict(self.usrlist_reverse)

    def getID(self, usr):
        try:
            return self.usrdict[usr]
        except:
            return -1
    
    def getUsr(self, id):
        try:
            return self.usrdict_reverse[id]
        except:
            return -1

class Prdlist(object):
    def __init__(self, filename, maxn=100000):
        lines = list(map(lambda x: x.split(), open(filename, "r", encoding='utf-8').readlines()[:maxn]))
        self.size = len(lines)

        self.prdlist = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.prddict = dict(self.prdlist)
        self.prdlist_reverse = [(item[1], item[0][0]) for item in zip(lines, range(self.size))]
        self.prddict_reverse = dict(self.prdlist_reverse)

    def getID(self, prd):
        try:
            return self.prddict[prd]
        except:
            return -1

    def getPrd(self, id):
        try:
            return self.prddict_reverse[id]
        except:
            return -1