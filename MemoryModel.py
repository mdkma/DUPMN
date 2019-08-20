#-*- coding: UTF-8 -*-  
import datetime, os
import pprint
import tensorflow as tf
# import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

class MemoryModel(object):
    def __init__(self, config, u, mem, emb):
        self.mem = mem
        self.emb = emb
        self.nhop = config.nhop
        self.sdim = config.sdim
        self.edim = config.edim
        self.batch_size = config.batch_size
        self.mem_size = config.mem_size
        self.init_std = config.init_std
        self.classes = config.classes

        self.u = u
        # C = tf.Variable(tf.truncated_normal([self.batch_size, self.edim]))
        # self.u = u * C

        # Construct input memory: Ain
        Ain = tf.nn.embedding_lookup(self.emb, self.mem)
        Bin = Ain
        # A = tf.Variable(tf.truncated_normal([self.batch_size, self.mem_size, self.sdim]))
        # B = tf.Variable(tf.truncated_normal([self.batch_size, self.mem_size, self.sdim]))
        # Ain = A * mem_docs
        # Bin = B * mem_docs
        # Ain = tf.tile(tf.expand_dims(u, axis=0), [self.batch_size,1,1])
        # Output memory: Bin
        # Bin = tf.Variable(tf.truncated_normal([self.batch_size, self.mem_size, self.sdim]))
        # Bin = tf.tile(tf.expand_dims(u, axis=0), [self.batch_size,1,1])
        # Bin = tf.nn.embedding_lookup(self.emb,self.mem)

        # Set current memroy if the input to memroy is less than mem_size > can be dynamic
        self.current_mem_size = self.mem_size
        u_list = []
        u_list.append(self.u)

        for h in range(self.nhop):
            u3dim = tf.reshape(u_list[-1], [-1, 1, self.sdim])
            # Calculate probability
            Aout = tf.matmul(u3dim, Ain, adjoint_b=True)
            Aout2dim = tf.reshape(Aout, [-1, self.current_mem_size])
            self.P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(self.P, [-1, 1, self.current_mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.sdim]) # o

            Dout = tf.add(Bout2dim, u_list[-1]) # sum: o+u

            # if self.lindim == self.edim:
            #     u_list.append(Dout)
            # elif self.lindim == 0:
            #     u_list.append(tf.nn.relu(Dout))
            # else:
            #     F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
            #     G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
            #     K = tf.nn.relu(G)
            #     u_list.append(tf.concat(axis=1, values=[F, K]))
            u_list.append(Dout)

        # 7-FINAL PREDICTION LAYER :: adjacent weight approach chosen
        self.W = tf.Variable(tf.random_normal([self.sdim, self.classes], stddev=self.init_std))
        # self.bias = tf.Variable(tf.constant(0.1, shape=[self.classes]))
        self.combineWeight = tf.Variable(tf.constant(0.5))
        self.pred_before_softmax = tf.matmul(u_list[-1], self.W)*self.combineWeight
        # final prediction is the self.prediction
        self.pred = tf.nn.softmax(self.pred_before_softmax)