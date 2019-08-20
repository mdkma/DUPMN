#-*- coding: UTF-8 -*-  
import datetime, os
import pprint
import tensorflow as tf

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

        # Construct input memory: Ain
        Ain = tf.nn.embedding_lookup(self.emb, self.mem)
        Bin = Ain

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
            u_list.append(Dout)

        # 7-FINAL PREDICTION LAYER :: adjacent weight approach chosen
        self.W = tf.Variable(tf.random_normal([self.sdim, self.classes], stddev=self.init_std))
        self.combineWeight = tf.Variable(tf.constant(0.5))
        self.pred_before_softmax = tf.matmul(u_list[-1], self.W)*self.combineWeight
        # final prediction is the self.prediction
        self.pred = tf.nn.softmax(self.pred_before_softmax)