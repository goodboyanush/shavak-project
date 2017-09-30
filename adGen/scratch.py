from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import string
import os
import time
import argparse
import json

from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# We disable pylint because we need python3 compatibility.

# linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access
# train script
from vocabulary import *
from S2SAttentionModel import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell


# from skip_wrapper import getRepr


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
	"""Get a loop_function that extracts the previous symbol and embeds it.
  
	Args:
	  embedding: embedding tensor for symbols.
	  output_projection: None or a pair (W, B). If provided, each fed previous
		output will first be multiplied by W and added B.
	  update_embedding: Boolean; if False, the gradients will not propagate
		through the embeddings.
  
	Returns:
	  A loop function.
	"""

	def loop_function(prev, _):
		if output_projection is not None:
			prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
		prev_symbol = math_ops.argmax(prev, 1)
		# Note that gradients will not propagate through the second parameter of
		# embedding_lookup.
		emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
		if not update_embedding:
			emb_prev = array_ops.stop_gradient(emb_prev)
		return emb_prev

	return loop_function


def rnn_decoder(initial_state,
                cell,
                loop_function=None,
                scope=None):
	"""RNN decoder for the sequence-to-sequence model.
  
	Args:
	  initial_state: 2D Tensor with shape [batch_size x cell.state_size].
	  cell: core_rnn_cell.RNNCell defining the cell function and size.
	  loop_function: If not None, this function will be applied to the i-th output
		in order to generate the i+1-st input, and decoder_inputs will be ignored,
		except for the first element ("GO" symbol). This can be used for decoding,
		but also for training to emulate http://arxiv.org/abs/1506.03099.
		Signature -- loop_function(prev, i) = next
		  * prev is a 2D Tensor of shape [batch_size x output_size],
		  * i is an integer, the step number (when advanced control is needed),
		  * next is a 2D Tensor of shape [batch_size x input_size].
	  scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
  
	Returns:
	  A tuple of the form (outputs, state), where:
		outputs: A list of the same length as decoder_inputs of 2D Tensors with
		  shape [batch_size x output_size] containing generated outputs.
		state: The state of each cell at the final time-step.
		  It is a 2D Tensor of shape [batch_size x cell.state_size].
		  (Note that in some cases, like basic RNN cell or GRU cell, outputs and
		   states can be the same. They are different for LSTM cells though.)
	"""
	with variable_scope.variable_scope(scope or "rnn_decoder", reuse=True):
		# inp = embedding_ops.embedding_lookup(tf.get_variable(name='embeddings'),
		#                                      sentence_to_word_ids(" ", word_to_id, tokenizer='simple', prependGO=True))
		inp = tf.convert_to_tensor([embedding_ops.embedding_lookup(tf.get_variable(name='embeddings'), x) for x in
		                            [word_to_id['<GO>']] * args.batch_size])
		# print (tf.gather(inp, [1]))
		# inp = tf.gather(inp, [0])
		# print ("Input")
		# print (inp)
		num_decoder_symbols = tf.get_variable(name='vocabulary_size', dtype=dtypes.int32_ref)

	with variable_scope.variable_scope(scope or "rnn_decoder"):
		state = initial_state
		outputs = []
		prev = None
		print(initial_state)

		for i in xrange(20):
			if prev is not None:
				with variable_scope.variable_scope("loop_function", reuse=True):
					inp = loop_function(prev, i)
				# print (i)
				# print (inp)
			if i > 0:
				variable_scope.get_variable_scope().reuse_variables()

			output, state = cell(inp, state)
			outputs.append(output)
			# print(output)
			if loop_function is not None:
				prev = output

	# print (outputs)

	return outputs, state


def generate_inputs(input_data):
	return random.choice(input_data.keys())


# def inference(cell, encoder_inputs):
def inference(cell, input_data):
	encoder_inputs = next_batch(args.batch_size, input_data)
	with tf.variable_scope("seq2seq", reuse=True) as scope:
		projection_B = tf.get_variable(name="proj_b")
		projection_W = tf.get_variable(name="proj_w")
		output_projection = (projection_W, projection_B)
		encoder_cell = copy.deepcopy(cell)

	with tf.variable_scope("seq2seq") as scope:
		embedding_matrix = tf.get_variable('embeddings', initializer=tf.truncated_normal(
			[vocabulary_size, args.embedding_size], mean=0.0, stddev=0.01),
		                                   trainable=bool(args.train_embedding_matrix))
		# print (encoder_inputs)
		emb_encoder_inputs = [embedding_ops.embedding_lookup(embedding_matrix, x) for x in
		                      encoder_inputs]
		emb_encoder_inputs = tf.unstack(emb_encoder_inputs, axis=1)
		print(emb_encoder_inputs)
		_, encoder_state = core_rnn.static_rnn(encoder_cell, emb_encoder_inputs, dtype=dtypes.float32)

		loop_function = _extract_argmax_and_embed(embedding_matrix, output_projection)

		outputs, state = rnn_decoder(encoder_state, cell, loop_function=loop_function, scope=scope)

		# print(outputs)
		# outputs = outputs[:-1]
		outputs = [tf.nn.xw_plus_b(o, projection_W, projection_B) for o in outputs]
	# print(outputs)
	return outputs


def prediction(logits):
	"""
	logits: A list of the length max_summary_seq_length of 2D Tensors with
					shape [batch_size x num_decoder_symbols] containing the generated
					outputs.
	Returns:
			predictions: 3D Tensor of float32 of size [batch_size, max_summary_seq_length, num_decoder_symbols]
	"""
	print('Inside prediction')
	logits = tf.stack(logits, axis=1)
	with tf.variable_scope("seq2seq", reuse=True) as scope:
		vocabulary_size = tf.get_variable(name='vocabulary_size', dtype=dtypes.int32_ref)
	logits_flat = tf.reshape(logits, [-1, vocabulary_size])
	predictions_flat = tf.nn.softmax(logits_flat)
	return tf.reshape(predictions_flat, tf.shape(logits))


def loss(logits, train_data, id_to_word):
	logits = tf.stack(logits, axis=1)
	pred = tf.argmax(input=logits, axis=2)

	sentence = []

	# print (pred)
	# to make it iterable
	pred = [tf.unstack(t) for t in tf.unstack(pred)]

	# since int32 is not supported

	keys = [tf.to_int64(x) for x in id_to_word.keys()]
	table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, id_to_word.values()), "NaN")
	# print (pred)
	for b in pred:
		sent = []
		for word in b:
			sent.append(table.lookup(word))
		# sent = tf.string_join(sent)
		sentence.append(sent)

	# sentence = tf.string_join(sentence)
	# repr = getRepr(sentence)

	return tf.random_normal([1]), table


def next_batch(batch_size, train_data):
	enc_new = []
	# TODO
	for x in xrange(args.batch_size):
	# enc_new.append([sentence_to_word_ids(generate_inputs(input_data=train_data), word_to_id, tokenizer='simple',
	#                                      prependGO=True)[1]])
	return enc_new


def getDist(train_data):
	dist = []
	for k in train_data.keys():
		genre = train_data[k]


def train(args, train_data, id_to_word=None, vocabulary_size=None, word_to_id=None, encoder_inputs=None):
	step = 0

	# save the args and the vocabulary

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
		f.write(json.dumps(vars(args), indent=1))

	with open(os.path.join(args.save_dir, 'word_to_id.json'), 'w') as f:
		f.write(json.dumps(word_to_id, indent=1))

	with open(os.path.join(args.save_dir, 'id_to_word.json'), 'w') as f:
		f.write(json.dumps(id_to_word, indent=1))
	target_vocab_size = len(word_to_id)

	# assign model type
	if args.model == 'rnn':
		cell = rnn_cell.BasicRNNCell(args.rnn_size)
	elif args.model == 'gru':
		cell = rnn_cell.GRUCell(args.rnn_size)
	elif args.model == 'basic_lstm':
		cell = rnn_cell.BasicLSTMCell(args.rnn_size)
	elif args.model == 'lstm':
		cell = rnn_cell.LSTMCell(args.rnn_size)
	else:
		raise Exception("model type not supported: {}".format(args.model))

	with tf.Graph().as_default():
		# sess = tf.InteractiveSession()
		cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=args.rnn_dropout_keep_prob)
		enc_new = tf.placeholder(tf.int32, name="enc_inputs")
		with tf.variable_scope("seq2seq") as scope:
			projection_B = tf.get_variable(name="proj_b", shape=[target_vocab_size])
			projection_W = tf.get_variable(name="proj_w", shape=[args.rnn_size, target_vocab_size])

			output_projection = (projection_W, projection_B)
			vocabulary_size = tf.get_variable(name='vocabulary_size', initializer=vocabulary_size)
			enc_new = []

		# input data generation
		# for x in xrange(args.batch_size):
		#     enc_new.append([sentence_to_word_ids(generate_inputs(input_data=train_data), word_to_id, tokenizer='simple', prependGO=True)[1]])

		# embedding_matrix = tf.get_variable('embeddings', initializer=tf.truncated_normal(
		#     [target_vocab_size, args.embedding_size], mean=0.0, stddev=0.01),
		# #                                    trainable=bool(args.train_embedding_matrix))
		# emb_encoder_inputs = [embedding_ops.embedding_lookup(embedding_matrix, args.genre)]
		# print(enc_new)
		# logits = inference(cell, encoder_inputs=enc_new)
		logits = inference(cell, train_data)
		predictions = prediction(logits)

		# print(logits)
		# print(predictions)

		loss_val, table = loss(logits, train_data=train_data, id_to_word=id_to_word)

		# directory to dump the intermediate models
		checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, 'checkpoints'))
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

		# location to dump the predictions for validation
		if not os.path.exists(os.path.join(args.save_dir, 'predictions')):
			os.makedirs(os.path.join(args.save_dir, 'predictions'))

		# summary directry
		summary_dir = os.path.abspath(os.path.join(args.save_dir, 'predicted_summaries'))
		if not os.path.exists(summary_dir):
			os.makedirs(summary_dir)

		saver = tf.train.Saver(tf.global_variables(), max_to_keep=max(5, args.early_stopping_batch_window + 2))

		session_conf = tf.ConfigProto(allow_soft_placement=True)
		session_conf.gpu_options.allow_growth = True
		with tf.Session(config=session_conf) as sess:
			# run the op to initialize the variables
			init = tf.global_variables_initializer()
			sess.run(init)
			sess.run(table.init)
			print('Trainable Variables')
			print('\n'.join([v.name for v in tf.trainable_variables()]))

			# instantiate a SummaryWriter to output summaries and the graph
			# train_summary_writer = tf.train.SummaryWriter(train_summary_dir, graph_def=sess.graph_def)
			# valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, graph_def=sess.graph_def)

			step = 0
			previous_epoch = 0

			epochs_completed = 0
			while epochs_completed < args.max_epochs:
				step += 1
				epochs_completed += 1
				start_time = time.time()
				# batch_enc_inputs, batch_enc_weights, batch_dec_weights = next_batch(args.batch_size, train_data)

				enc_new = []

				# input data generation
				# for x in xrange(args.batch_size):
				#     enc_new.append([sentence_to_word_ids(generate_inputs(input_data=train_data), word_to_id, tokenizer='simple', prependGO=True)[1]])
				# batch_enc_inputs = enc_new
				# feed_dict = {enc_new: batch_enc_inputs}

				# _, loss_value = sess.run([logits, loss_val], feed_dict=feed_dict)
				_, loss_value = sess.run([logits, loss_val])
				duration = time.time() - start_time

				if epochs_completed > previous_epoch:
					previous_epoch = epochs_completed
					if epochs_completed % args.save_every_epochs == 0 or epochs_completed == args.max_epochs - 1:
						path = saver.save(sess, checkpoint_prefix, global_step=epochs_completed)
						print("Saved model checkpoint to {}".format(path))


def load_summaries(file_name):
	print('Reading file ' + file_name)
	data = np.load(file_name, encoding='ASCII').item()
	return data


# def load_data(train_file, vocabulary_min_count):
# 	data = load_summaries(train_file)
# 	print('Loaded data...building vocab')
# 	vocabulary_sentences = []
#
# 	for sent in data.values():
# 		vocabulary_sentences.extend(sent)
#
# 	for k in data.keys():
# 		# data[k] = [sent_tokenize(para) for para in data[k]]
# 		temp = []
# 		for para in data[k]:
# 			temp.extend(sent_tokenize(para.decode('utf-8')))
# 		data[k] = temp
# 		# print (data[k])
# 		# data[k] = [sent.translate(translate_table) for sent in data[k]]
#
# 		temp = []
# 		for sent in data[k]:
# 			if len(word_tokenize(sent)) > 15:
# 				temp.extend(' '.join(word_tokenize(sent)[:15]))
# 			else:
# 				temp.extend(sent)
# 		data[k] = temp
# 	print(len(data['War']))
# 	word_to_id, id_to_word = build_vocabulary(vocabulary_sentences, min_count=vocabulary_min_count, tokenizer='simple')
# 	with open('word_to_id.txt', 'w+') as wf:
# 		wf.write(str(word_to_id))
#
# 	return data, id_to_word, word_to_id


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--encoder_embedding_size', type=int, default=20,
	                    help='size of the embeddings for records (<65)')
	parser.add_argument('--decoder_embedding_size', type=int, default=64,
	                    help='size of the embeddings for output words')  # 100,200 32,64,128
	parser.add_argument('--bidirectional_encoder', type=int, default=0,
	                    help='use bidirectional rnn for encoder or not (1 or 0)')
	parser.add_argument('--rnn_size', type=int, default=64, help='size of RNN hidden state')  # 100
	parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN (default: 1)')
	parser.add_argument('--model', type=str, default='gru', help='rnn, gru, basic_lstm, or lstm (default: gru)')
	parser.add_argument('--rnn_dropout_keep_prob', type=float, default=1.0,
	                    help='dropout keep probability when using multiple RNNs')
	parser.add_argument('--output_dropout_keep_prob', type=float, default=1.0,
	                    help='dropout keep probability for the output (i.e. before the softmax layer)')
	parser.add_argument('--train_embedding_matrix', type=int, default=1,
	                    help='if 0 does not train the embedding matrix and keeps it fixed')
	parser.add_argument('--use_pretrained_embedding_matrix', type=int, default=0,
	                    help='if 1 use the pre-trained word2vec for initializing the embedding matrix')
	parser.add_argument('--pretrained_embedding_filename', type=str,
	                    default='../resources/GoogleNews-vectors-negative300.bin',
	                    help='full path to the .bin file containing the pre-trained word vectors')
	parser.add_argument('--embedding_size', type=int, default=100,
	                    help='size of embedding if pretrained embeddings are not used')

	# Training parameters
	parser.add_argument('--batch_size', type=int, default=1024, help='batch size')  # 16, 32, 64
	parser.add_argument('--max_sent_len', type=int, default=20, help='max sentence length generated')
	parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs')
	parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam')  # rmsprop, 0.01, 0.1
	parser.add_argument('--max_gradient_norm', type=float, default=1000.0,
	                    help='maximum gradient norm for clipping. currently unused.')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--save_dir', type=str, default='exp_1', help='directory to save checkpointed models')
	parser.add_argument('--print_every', type=int, default=1, help='print some info after this many batches')
	parser.add_argument('--summary_every', type=int, default=10,
	                    help='dump summaries for tensorboard after this many batches')
	parser.add_argument('--save_every_epochs', type=int, default=1, help='save the model after this many epochs')
	parser.add_argument('--evaluate_every_epochs', type=int, default=1,
	                    help='evaluate the model on the entire data after this many epochs')

	parser.add_argument('--early_stopping', type=int, default=1,
	                    help='if 1 enables early stopping based on loss on the validation split')
	parser.add_argument('--early_stopping_batch_window', type=int, default=5,
	                    help='early stop if the validation loss is greater than the loss from these many previous steps')
	parser.add_argument('--early_stopping_threshold', type=float, default=0.05, help='threshold for early stopping')

	# Task specific
	parser.add_argument('--trainFilename', type=str, default=None, help='name of the train file')
	parser.add_argument('--validFilename', type=str, default=None, help='name of the valid file')
	parser.add_argument('--testFilename', type=str, default=None, help='name of the test file')
	parser.add_argument('--genre', type=str, default=None, help='Name of the genre')
	parser.add_argument('--vocab_min_count', type=int, default=1, help='keep words whose count is >= vocab_min_count')

	args = parser.parse_args()

	# train_data, test_data, valid_data, id_to_word = load_data(args.trainFilename, args.testFilename, args.validFilename, args.max_q, args.max_t, args.vocab_min_count)
	# data, id_to_word, word_to_id = load_data(args.trainFilename, args.vocab_min_count)
	# print(id_to_word)
	vocabulary_size = len(id_to_word)
	# train(args, train_data, valid_data, id_to_word=id_to_word, vocabulary_size=vocabulary_size)
	train(args, data, id_to_word=id_to_word, vocabulary_size=vocabulary_size, word_to_id=word_to_id,
	      encoder_inputs=[args.genre])
