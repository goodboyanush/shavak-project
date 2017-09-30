import tensorflow as tf    
import numpy as np
import argparse
import time
import datetime
import os
import json
import traceback
import math
from attention_seq2seq import *
from sequence_loss import *
import tensorflow.contrib.rnn as rnn_cell
#import contrib.rnn_cell as rnn_cell

__author__  = "Parag Jain"
__email__   = "pajain06@in.ibm.com"

class S2SAttentionModel(object):
  def __init__(self,
  	          target_vocab_size,
  	          rnn_size,
  	          num_layers,
  	          max_gradient_norm,
  	          max_summary_seq_length,
  	          model,
              batch_size,
              embedding_size,
              initial_embedding=None,
              embedding_trainable = True,
  	          learning_rate = 0.001,
  	          optimizer = 'adam',
  	          forward_only=False,
  	          rnn_dropout_keep_prob=1.0,
  	          output_dropout_keep_prob=1.0,
  	          dtype=tf.float32):

    self.target_vocab_size = target_vocab_size
    self.rnn_size = rnn_size
    self.num_layers = num_layers
    self.max_gradient_norm = max_gradient_norm
    self.max_summary_seq_length = max_summary_seq_length
    self.model = model
    self.learning_rate = learning_rate
    self.optimizer = optimizer
    self.forward_only = forward_only
    self.rnn_dropout_keep_prob = rnn_dropout_keep_prob
    self.output_dropout_keep_prob = output_dropout_keep_prob
    self.dtype = dtype
    self.projection_B = tf.get_variable(name="proj_b", shape=[target_vocab_size])
    self.projection_W = tf.get_variable(name="proj_w", shape=[rnn_size, target_vocab_size])
    self.embedding_size = embedding_size
    self.initial_embedding = initial_embedding
    self.embedding_trainable = embedding_trainable

    # choose the RNN cell
    if self.model == 'rnn':
      self.cell = rnn_cell.BasicRNNCell(self.rnn_size)
    elif self.model == 'gru':
      self.cell = rnn_cell.GRUCell(self.rnn_size)
    elif self.model == 'basic_lstm':
      self.cell = rnn_cell.BasicLSTMCell(self.rnn_size)
    elif self.model == 'lstm':
      self.cell = rnn_cell.LSTMCell(self.rnn_size)
    else:
      raise Exception("model type not supported: {}".format(self.model))  
    #Add dropout to RNN Cell
    self.cell = rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.rnn_dropout_keep_prob)
    #Make RNN Cell multi layer
    if num_layers > 1:
        self.cell = rnn_cell.MultiRNNCell([self.cell] * num_layers)


  def inference(self, encoder_inputs, decoder_inputs):
    print('Inference')
    print('Encoder input shape')
    print(encoder_inputs.get_shape())
    encoder_inputs = tf.unpack(encoder_inputs, axis=1)
    decoder_inputs = tf.unpack(decoder_inputs, axis=1)
    print('Length of encoder input after unpack, should be seq length each of size batch size')
    print(len(encoder_inputs))
    output_projection=(self.projection_W, self.projection_B)
    
    with tf.variable_scope("embedding_attention_seq2seq") as scope:
      if self.initial_embedding is not None:
        self.embedding_matrix = tf.get_variable('embeddings',initializer=self.initial_embedding, trainable=self.embedding_trainable) 
      else:
        self.embedding_matrix = tf.get_variable('embeddings', initializer=tf.truncated_normal([self.target_vocab_size, self.embedding_size], mean=0.0, stddev=0.01), trainable=self.embedding_trainable)

    outputs, state = embedding_attention_seq2seq(encoder_inputs, decoder_inputs,
      self.cell,num_encoder_symbols=self.target_vocab_size, num_decoder_symbols=self.target_vocab_size, embedding_size=self.embedding_size,
      output_projection=output_projection)

    print('Inference output size')
    print(len(outputs))
    print(outputs[0].get_shape())
    outputs = outputs[:-1]
    outputs = [tf.matmul(o, self.projection_W) + self.projection_B for o in outputs]
    return outputs

  def prediction(self, logits):
    '''
    logits: A list of the length max_summary_seq_length of 2D Tensors with
                    shape [batch_size x num_decoder_symbols] containing the generated
                    outputs.
    Returns:
            predictions: 3D Tensor of float32 of size [batch_size, max_summary_seq_length, num_decoder_symbols]
    '''
    print('Inside prediction')
    logits = tf.pack(logits, axis=1)
    logits_flat = tf.reshape(logits, [-1, self.target_vocab_size])
    predictions_flat = tf.nn.softmax(logits_flat)
    return tf.reshape(predictions_flat, tf.shape(logits))

  def loss(self, logits, decoder_inputs, target_weights):
    '''
    logits
    '''
    print('Inside loss')
    decoder_inputs = tf.unpack(decoder_inputs, axis=1)
    targets = [decoder_inputs[i+1] for i in xrange(len(decoder_inputs) - 1)]
    target_weights = tf.unpack(target_weights, axis=1)
    print(len(decoder_inputs))
    print(decoder_inputs[0].get_shape())
    print(len(targets))
    print(targets[0].get_shape())
    print(len(target_weights))
    print(target_weights[0].get_shape())
    print('logit shape')
    print(len(logits))
    print(logits[0].get_shape())
    return tf.nn.seq2seq.sequence_loss(logits, targets, target_weights)

  def training(self, loss):
    """ sets up the training ops
        params:
            loss: loss tensor, from loss()
    """

    # create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = 'adam'
    learning_rate = self.learning_rate
    # create the gradient descent optimizer with the given learning rate.
    if optimizer == 'gradient_descent':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5)
    else:
      raise Exception("optimizer type not supported: {}".format(optimizer))
      
    # use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    #train_op = optimizer.minimize(loss, global_step=global_step)

    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op

  def evaluation(self, loss):
    '''
    Returns perplexity
    '''
    return tf.to_float(math.exp(float(loss)) if loss < 300 else float("inf"))


