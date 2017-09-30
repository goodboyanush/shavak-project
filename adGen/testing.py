from scratch import inference, prediction
import tensorflow as tf
import numpy as np


def getSentence(genre):
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         # embeddings = np.load("/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy")
    #         # dir(tf.contrib)
    #         # saver = tf.train.import_meta_graph("/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424.meta")
    #         saver = tf.train.Saver()
    #         saver.restore(sess, tf.train.latest_checkpoint("/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/."))
    #         # print tf.train.Saver.restore(sess, "/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424")
    #         print "hello"
    sess = tf.InteractiveSession()
    # layer1_weightseights = tf.Variable(tf.truncated_normal(
    #     [patch_size, patch_size, num_channels, depth], stddev=0.1), name="layer1_weights")


    tf.train.Saver().restore(sess, "/data/anussank/shavak/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424")

if __name__ == '__main__':
    getSentence("War")