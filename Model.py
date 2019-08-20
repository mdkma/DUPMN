#-*- coding: UTF-8 -*-  
import datetime, os
import pprint
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell, dynamic_rnn
import numpy as np
import math
# import cPickle
import pickle
from MemoryModel import MemoryModel
from LSTMModel import LSTMModel

pp = pprint.PrettyPrinter()

class Model(object):
    def __init__(self, config, sess, wordVectors, train_data, test_data):
        self.config = config
        self.word_vectors = wordVectors
        self.classes = config.classes
        self.iterations = config.iterations
        self.batch_size = config.batch_size
        self.edim = config.edim
        self.sdim = config.sdim
        self.nhop = config.nhop
        self.mem_size = config.mem_size
        self.init_std = config.init_std
        self.current_lr = config.init_lr
        self.lindim = config.lindim
        self.checkpoint_dir = config.checkpoint_dir
        self.is_test = config.is_test
        self.show = config.show
        self.data_dir = config.data_dir
        self.data_name = config.data_name
        self.design = config.design

        self.doc_emb_method = config.doc_emb_method
        self.dropout_keep_prob = config.dropout_keep_prob

        self.debug = config.debug # debug
        self.sess = sess
        self.word_vectors_dimen = self.word_vectors.shape[1]
        self.log_dir = '../log/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt'

        # init document embedding
        if self.doc_emb_method == 'generate':
            # init brand new embedding for document and testing data
            self.doc_emb = np.empty(shape=[train_data.num_doc,self.edim], dtype=np.float32)
            self.doc_emb_test = np.empty(shape=[test_data.num_doc,self.edim], dtype=np.float32)
        else:
            # load pre-trained document embedding
            self.doc_emb = np.asarray(self.load_doc_emb(), dtype=np.float32)
            self.doc_emb_test = np.asarray(self.load_doc_emb_test(), dtype=np.float32)
            print('-> loaded doc emb for training and testing data')
            # print type(self.doc_emb)
            print(np.shape(self.doc_emb))
            print(np.shape(self.doc_emb_test))

        # init final prediction result
        self.pred_test = np.empty(shape=[test_data.num_doc,], dtype=np.float32)

        # each batch has 32 document, N sentences
        # input_data should be in shape [numSentences, maxNumWords]
        self.input_data = tf.placeholder(tf.int32, [None, None], name="input")
        self.word_vecs = tf.placeholder(tf.int32, [None, None], name="word_vecs")
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.classes], name="labels")
        self.wordmask = tf.placeholder(tf.float32, [None, None], name="wordmask")
        self.sentencemask = tf.placeholder(tf.float32, [None, None], name="sentencemask")

        # Placeholder for related docs about user and product
        self.main_docs = tf.placeholder(tf.float32, [self.batch_size, self.edim], name="main_docs")
        self.usr_docs = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="usr_docs") # batch_size * mem_size
        self.prd_docs = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="prd_docs")

        # Get dynamic shape
        self.numSentences = tf.shape(self.input_data)[0]
        self.maxNumWords = tf.shape(self.input_data)[1]
        
        self.params = []

    def build_graph_mn(self):
        # EXTERNAL MEMORY
        # Embed the main input document
        u = self.main_docs
        if self.doc_emb_method == 'preload_update':
            u = self.doc_representation
        if self.doc_emb_method == 'preload_no_update':
            self.doc_representation = self.main_docs
        self.current_mem_size = self.mem_size

        if self.design == 'one':
            ''' One memory: only use user or product memory network, rather than both '''
            m = MemoryModel(self.config, u, self.prd_docs, self.doc_emb)
            self.prediction_0 = m.pred_before_softmax
            self.prediction = m.pred
            self.params = [m.W]

        elif self.design == 'two':
            ''' separate two memory: one for user, one for product '''
            m_usr = MemoryModel(self.config, u, self.usr_docs, self.doc_emb)
            m_prd = MemoryModel(self.config, u, self.prd_docs, self.doc_emb)
            self.params = [m_usr.W, m_prd.W, m_usr.combineWeight, m_prd.combineWeight]
            self.weight_usr = m_usr.combineWeight
            self.weight_prd = m_prd.combineWeight

            # Combine Two Network
            self.prediction_0 = tf.add(m_usr.pred_before_softmax, m_prd.pred_before_softmax)
            self.prediction = tf.nn.softmax(self.prediction_0)

    def build_graph_opt(self):
        self.correctPred = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.labels,1))
        self.temp1 = tf.cast(tf.argmax(self.prediction,1), tf.float32)
        self.temp2 = tf.cast(tf.argmax(self.labels,1), tf.float32)
        self.mse = tf.losses.mean_squared_error(self.temp2, self.temp1)
        # self.mae, _ = tf.metrics.mean_absolute_error(self.temp2, self.temp1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction_0, labels=self.labels))
        self.optim = tf.train.AdamOptimizer(self.current_lr).minimize(self.loss)
        # self.opt = tf.train.GradientDescentOptimizer(self.current_lr)
        # self.optim = tf.train.GradientDescentOptimizer(self.current_lr).minimize(self.loss)

        # grads_and_vars = self.opt.compute_gradients(self.loss, self.params)
        # clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], 50), gv[1]) \
        #                            for gv in grads_and_vars]

        # inc = self.global_step.assign_add(1)
        # with tf.control_dependencies([inc]):
        # self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, inputs):
        # run through all batches
        #print("A")
        if self.debug:
            train_size = 100
        else:
            train_size = inputs.epoch
        batchList = np.random.randint(inputs.epoch-1, size = train_size) # do not choose last batch
        n = 0
        cost = 0
        accuracy_total = 0
        total_weight_prd = 0
        total_weight_usr = 0
        #print("B")

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('train', max=train_size)

        for i in batchList:
            # Load next batch into Dataset class instance: inputs
            # inputs.gene_batch(i)
            indx = i
            #print(i)

            nextBatchData = np.array(inputs.docs[indx]).astype(np.int32).transpose()
            # Make the labels one-hot
            labels_temp = inputs.label[indx]
            nextBatchLabels = np.array(np.eye(self.classes)[labels_temp], dtype=np.float32)      
            nextWordMask = np.array(inputs.wordmask[indx]).astype(np.float32).transpose()
            nextSentenceMask = np.array(inputs.sentencemask[indx]).astype(np.float32).transpose()
            nextUsrDocs = inputs.gene_usr_context(inputs.usr[indx], self.mem_size)
            nextPrdDocs = inputs.gene_prd_context(inputs.prd[indx], self.mem_size)
            nextMainDocs = self.doc_emb[i*self.batch_size:(i+1)*self.batch_size,:]

            # print (np.shape(nextBatchData))
            # print (np.shape(nextBatchLabels))
            # print (np.shape(nextWordMask))
            # print (np.shape(nextSentenceMask))
            # print (np.shape(nextUsrDocs))
            # print (np.shape(nextPrdDocs))
            # print (np.shape(nextMainDocs))

            _, _loss, _accuracy, _doc_representation, _weight_usr, _weight_prd = self.sess.run(
                [self.optim, self.loss, self.accuracy, self.doc_representation, self.weight_usr, self.weight_prd],
                feed_dict={
                    self.input_data: nextBatchData,
                    self.word_vecs: self.word_vectors,
                    self.labels: nextBatchLabels,
                    self.wordmask: nextWordMask,
                    self.sentencemask: nextSentenceMask,
                    self.main_docs: nextMainDocs,
                    self.usr_docs: nextUsrDocs,
                    self.prd_docs: nextPrdDocs
                })
            cost += _loss
            accuracy_total += _accuracy
            total_weight_prd += _weight_prd
            total_weight_usr += _weight_usr
            if self.doc_emb_method != 'preload_no_update':#update document representation
                self.doc_emb[i*self.batch_size:(i+1)*self.batch_size,:] = _doc_representation
            # print '- count: ', n, '\t - batch number: ', i, '\t - loss: ', _loss

            n += 1
            if self.show: bar.next()

        if self.show: bar.finish()
        return cost/train_size, accuracy_total/train_size, total_weight_usr/train_size, total_weight_prd/train_size

    def test(self, inputs):
        if self.debug:
            test_size = 100
        else:
            test_size = inputs.epoch
        batchList = np.random.randint(inputs.epoch-1, size = test_size) # inputs.epoch do not choose last batch
        n = 0
        cost = 0
        accuracy_total = 0
        mae_total = 0
        mse_total = 0

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('test ', max=test_size)

        for i in batchList:
            # Load next batch into Dataset class instance: inputs
            # inputs.gene_batch(i)
            indx = i

            nextBatchData = np.array(inputs.docs[indx]).astype(np.int32).transpose()
            # Make the labels one-hot
            labels_temp = inputs.label[indx]
            nextBatchLabels = np.array(np.eye(self.classes)[labels_temp], dtype=np.float32)
            nextWordMask = np.array(inputs.wordmask[indx]).astype(np.float32).transpose()
            nextSentenceMask = np.array(inputs.sentencemask[indx]).astype(np.float32).transpose()
            nextUsrDocs = inputs.gene_usr_context(inputs.usr[indx], self.mem_size)
            nextPrdDocs = inputs.gene_prd_context(inputs.prd[indx], self.mem_size)
            nextMainDocs = self.doc_emb_test[i*self.batch_size:(i+1)*self.batch_size,:]

            _loss, _accuracy, _correctPred, _prediction, _pred1, _label1, _mse, _doc_representation = self.sess.run(
                [self.loss, self.accuracy, self.correctPred, self.prediction, self.temp1, self.temp2, self.mse, self.doc_representation],
                feed_dict={
                    self.input_data: nextBatchData,
                    self.word_vecs: self.word_vectors,
                    self.labels: nextBatchLabels,
                    self.wordmask: nextWordMask,
                    self.sentencemask: nextSentenceMask,
                    self.main_docs: nextMainDocs,
                    self.usr_docs: nextUsrDocs,
                    self.prd_docs: nextPrdDocs
                })
            if self.doc_emb_method != 'preload_no_update':#update document representation
                self.doc_emb_test[i*self.batch_size:(i+1)*self.batch_size,:] = _doc_representation
            # save final prediction result here to self.pred_test from self.temp1
            self.pred_test[i*self.batch_size:(i+1)*self.batch_size] = _pred1
            # print labels_temp
            # print _prediction
            # print _label1
            # print _pred1
            # print 'MSE:', _mse
            mae = np.sum(np.absolute(_pred1.astype("float") - _label1.astype("float")))/self.batch_size
            # print mae
            # print _correctPred
            # print '- count: ', n, '\t - batch number: ', i, '\t - loss: ', _loss
            # print 'accuracy: ', _accuracy
            cost += _loss
            accuracy_total += _accuracy
            mse_total += _mse
            mae_total += mae

            n += 1
            if self.show: bar.next()
        if self.show: bar.finish()
        return cost/test_size, accuracy_total/test_size, mae_total/test_size, np.sqrt(mse_total/test_size)

    def run(self, train_data, test_data):
        if not self.is_test:
            # TRAINING
            print ('.. Training start. epoch: %s' % train_data.epoch)
            bestAccuracy = 0
            bestIteration = 0

            record_file = open(self.log_dir,'w')
            record_file.write('epoch,learning_rate,test_loss,test_accuracy,test_mae,test_rmse,prev_best_accuracy,train_loss,train_accuracy,avg_weight_usr,avg_weight_prd\n')
            record_file.close()

            for idx in range(self.iterations):
                print ('-> iteration: %s' % idx)
                train_loss, train_accuracy, avg_weight_usr, avg_weight_prd = self.train(train_data)
                test_loss, test_accuracy, test_mae, test_rmse = self.test(test_data)
                state = {
                    '             epoch': idx,
                    '     learning_rate': self.current_lr,
                    '         test_loss': test_loss,
                    '     test_accuracy': test_accuracy,
                    '    train_accuracy': train_accuracy,
                    '        train_loss': train_loss,
                    '          test_mae': test_mae,
                    '         test_rmse': test_rmse,
                    'prev_best_accuracy': bestAccuracy,
                    '    avg_weight_usr': avg_weight_usr,
                    '    avg_weight_prd': avg_weight_prd,
                    '   prev_best_epoch': bestIteration
                }
                pp.pprint(state)

                with open(self.log_dir, "a") as record_file:
                    record_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (idx, self.current_lr, test_loss, test_accuracy, test_mae, test_rmse, bestAccuracy, train_loss, train_accuracy, avg_weight_usr, avg_weight_prd))

                if test_accuracy > bestAccuracy and idx < 6:
                    bestAccuracy = test_accuracy

                if test_accuracy > bestAccuracy and idx > 5:
                    bestAccuracy = test_accuracy
                    bestIteration = idx
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "Network.model"))
                    print('--> model saved: better accuracy!')
                    if self.doc_emb_method != 'preload_no_update':
                        self.save_doc_emb(self.doc_emb)
                        self.save_doc_emb_test(self.doc_emb_test)
                    self.save_pred_test(self.pred_test)

                # Learning rate annealing
                # if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                #     self.current_lr = self.current_lr / 1.5
                #     self.lr.assign(self.current_lr).eval()
                # if self.current_lr < 1e-5: break
        else:
            # TESTING
            self.load() # restore trained model
            print ('.. Testing start. epoch: %s' % test_data.epoch)

            test_loss = np.sum(self.test(test_data))

            state = {
                # 'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            pp.pprint(state)

    def load(self):
        # load whole model
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")

    def save_doc_emb(self, doc_emb):
        # TODO: change cPickle
        f = open('%s/%s/emb_doc_train.save' % (self.data_dir, self.data_name), 'wb')
        # cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
        pickle.dump(doc_emb, f)
        f.close()
        print('--> saved doc embedding: train_data')

    def save_doc_emb_test(self, doc_emb):
        f = open('%s/%s/emb_doc_test.save' % (self.data_dir, self.data_name), 'wb')
        # cPickle.dump(doc_emb, f, protocol=cPickle.HIGHEST_PROTOCOL)
        pickle.dump(doc_emb, f)
        f.close()
        print('--> saved doc embedding: test_data')

    def save_pred_test(self, result):
        f = open('%s/%s/pred_test_mn.save' % (self.data_dir, self.data_name), 'wb')
        # cPickle.dump(result, f, protocol=cPickle.HIGHEST_PROTOCOL)
        pickle.dump(result, f)
        f.close()
        print('--> saved final prediction: test_data')

    def load_doc_emb(self):
        f = open('%s/%s/emb_doc_train.save' % (self.data_dir, self.data_name), 'rb')
        result = pickle.load(f, encoding='latin1')
        f.close()
        return result

    def load_doc_emb_test(self):
        f = open('%s/%s/emb_doc_test.save' % (self.data_dir, self.data_name), 'rb')
        result = pickle.load(f, encoding='latin1')
        f.close()
        return result