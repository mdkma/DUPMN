# -*- coding: UTF-8 -*-  
import os, time
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import pprint
import pickle
import tensorflow as tf

from Dataset import *
from Model import Model

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("data_name", "yelp13", "data set name [yelp13]") # yelp14 IMDB
flags.DEFINE_integer("classes", 5, "number of classes in this dataset [5]")
flags.DEFINE_boolean("chinese", False, "True for chinese dataset for parsing sentence to words [False]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")

flags.DEFINE_integer("iterations", 500, "number of iterations [100]")
flags.DEFINE_string("design", "two", "how many memory [two]") # seperate
flags.DEFINE_string("doc_emb_method", "preload_no_update", "use existing document embedding, do not update [preload_no_update]") # generate(only use LSTM to train doc emb); preload_update(load existing, use lstm and mn); preload_no_update(just only memory network)
flags.DEFINE_integer("nhop", 1, "number of hops [3]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_float("init_lr", 0.0001, "initial learning rate [0.01]")
flags.DEFINE_integer("batch_size", 32, "how many documents in a batch [32]")
flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout_keep_prob for LSTM dropout layer [1.0]")

flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_integer("lindim", 200, "linear part of the state [75]")
# flags.DEFINE_boolean("linear_start", False, "remove softmax between each layer memory, only left last one [False]") # TODO: haven't do re-insert when validation loss stop decreasing

flags.DEFINE_integer("edim", 200, "depth of LSTM network, embedding dimension for sens and docs [200]")
flags.DEFINE_integer("sdim", 200, "internal state of memory network dimension [200]")
flags.DEFINE_string("data_dir", "../data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_boolean("show", True, "print progress [True]")
flags.DEFINE_boolean("load_saved_model", False, "load saved model [False]")
flags.DEFINE_boolean("ten2five_classes_train", False, "convert 10 classes labels to 5 classes [False]")
flags.DEFINE_boolean("ten2five_classes_test", False, "convert 10 classes labels to 5 classes [False]")

flags.DEFINE_boolean("debug", False, "whether i am debugging or not [False]")
FLAGS = flags.FLAGS

important_setting = {
        '     data_name': FLAGS.data_name,
        '       classes': FLAGS.classes,
        '       chinese': FLAGS.chinese,
        '        design': FLAGS.design,
        'doc_emb_method': FLAGS.doc_emb_method,
        '          nhop': FLAGS.nhop,
        '       init_lr': FLAGS.init_lr,
    }

def main(_):
    start_time = time.time()
    pp.pprint(important_setting)
    pp.pprint(flags.FLAGS.__flags)
    
    # Load vocabulary, usrdict and prddict
    voc = Wordlist('%s/%s/wordlist.txt' % (FLAGS.data_dir, FLAGS.data_name))
    usrdict = Usrlist('%s/%s/usrlist.txt' % (FLAGS.data_dir, FLAGS.data_name))
    prddict = Prdlist('%s/%s/prdlist.txt' % (FLAGS.data_dir, FLAGS.data_name))
    # Load pretrained word vectors
    f = open('%s/%s/embinit.save' % (FLAGS.data_dir, FLAGS.data_name), 'rb')
    wordVectors = pickle.load(f, encoding='latin1')
    f.close()
    
    # Load trainset and test set
    print ('.. loading datasets')
    if FLAGS.debug:
        train_data = Dataset('%s/%s/test.txt' % (FLAGS.data_dir, FLAGS.data_name), voc, usrdict, prddict, FLAGS.batch_size, ten2five_classes = FLAGS.ten2five_classes_test, chinese=FLAGS.chinese)
    else:
        train_data = Dataset('%s/%s/train.txt' % (FLAGS.data_dir, FLAGS.data_name), voc, usrdict, prddict, FLAGS.batch_size, ten2five_classes = FLAGS.ten2five_classes_train, chinese=FLAGS.chinese)
    print ('-> train data loaded')
    test_data = Dataset('%s/%s/test.txt' % (FLAGS.data_dir, FLAGS.data_name), voc, usrdict, prddict, FLAGS.batch_size, ten2five_classes = FLAGS.ten2five_classes_test, chinese=FLAGS.chinese)
    print ('-> test data loaded')

    # use memory network, load existing document embedding
    tf.reset_default_graph()
    print ('.................................. training memory network')
    with tf.Session() as sess:
        model = Model(FLAGS, sess, wordVectors, train_data, test_data)
        model.build_graph_mn()
        model.build_graph_opt()
        if FLAGS.load_saved_model:
            model.load()
            print ('-> loaded existing network model')
        model.run(train_data, test_data)

    end_time = time.time()
    print ('-> Done. The training takes about %s seconds' % (str(end_time - start_time)))

if __name__ == "__main__":
    tf.app.run()