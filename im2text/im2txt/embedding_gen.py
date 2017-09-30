import numpy as np
import cPickle as pkl
import tensorflow as tf
from collections import OrderedDict, defaultdict

EMBEDDINGS_PATH = "/data/anussank/shavak/neuralstorytelling/coco_embedding.npz"


def load_model():
    # Load the worddict
    with open('%s.dictionary.pkl'%EMBEDDINGS_PATH, 'rb') as f:
        worddict = pkl.load(f)

    data = np.load(EMBEDDINGS_PATH)

    #Loads the projection onto the 1024 space in the COCO set
    W = data['encoder_Wx']
    b = data['encoder_bx']
    Wemb = data['Wemb']

    model = {}
    model['worddict'] = worddict
    model['W'] = W
    model['b'] = b
    model['Wemb'] = Wemb

    return model

def getEmb(model, words):

    # quick check if a word is in the dictionary
    d = defaultdict(lambda: 0)
    for w in model['worddict'].keys():
        d[w] = 1

    seqs = []
    for w in words:
        seqs.append(model['worddict'][w] if d[w] > 0 and model['worddict'][w] < len(model['worddict']) else 1)

    emb = tf.constant(model['Wemb'], name='embed_gen')

    emb_curr = tf.nn.embedding_lookup(emb, seqs)

    # print (emb_curr)
    proj = tf.nn.xw_plus_b(emb_curr, model['W'], model['b'])

    return proj



