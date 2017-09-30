'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from vocabulary import build_vocabulary, getSentences
import PretrainedEmbeddings

def getSentences(data):
	sent = []
	for k in data.keys():
		for sent_ in data[k]:
			sent.append(sent_)

	return sent

def build_vocabulary(sentences, genres, min_count=1):
	""" build the vocabulary from a list of `sentences'
	uses word_tokenize from nltk for word tokenization

	:params:
		sentences: list of strings
			the list of sentences
		min_count: int
			keep words whose count is >= min_count                 

	:returns:
	   word_to_id: dict
			dict mapping a word to its id, e.g., word_to_id['the'] = 4
			the id start from 4
			3 is reserved for <GO> (in case of decoder RNN for En-Dec architecture)
			2 is reserved for out-of-vocabulary words (<OOV>)
			1 is reserved for end-of-sentence marker (<EOS>)
			0 is reserved for padding (<PAD>)
	"""

	wordcount = Counter()
	tokenizer = RegexpTokenizer(r'\w+')
	for sentence in sentences:
		tokens = tokenizer.tokenize(sentence)
		tokens = map(str.lower, tokens)
		wordcount.update(tokens)

	for k in genres:
		# map(str.lower, k)
		wordcount.update([k])

	print('vocabulary size = %d' % (len(wordcount)))

	# filtering
	count_pairs = wordcount.most_common()
	count_pairs = [c for c in count_pairs if c[1] >= min_count]

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(4, len(words) + 4)))
	print('vocabulary size = %d (after filtering with min_count =  %d)' % (len(word_to_id), min_count))

	word_to_id['<PAD>'] = 0
	word_to_id['<EOS>'] = 1
	word_to_id['<OOV>'] = 2
	word_to_id['<GO>'] = 3

	id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

	return word_to_id, id_to_word


def RNN(x, weights, biases, scope):
	# reshape to [1, n_input]
	x = tf.reshape(x, [-1, n_input])

	# Generate a n_input-element sequence of inputs
	# (eg. [had] [a] [general] -> [20] [6] [33])
	x = tf.split(x, n_input, 1)

	# 2-layer LSTM, each layer has n_hidden units.
	# Average Accuracy= 95.20% at 50k iter
	# rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
	rnn_cell = rnn.GRUCell(n_hidden)
	# 1-layer LSTM with n_hidden units but with lower accuracy.
	# Average Accuracy= 90.60% 50k iter
	# Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
	# rnn_cell = rnn.BasicLSTMCell(n_hidden)

	# generate prediction
	outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32, scope=scope)

	# there are n_input outputs but
	# we only want the last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

def read_data(fname):
	# with open(fname) as
	# 	content = f.readlines()
	# content = [x.strip() for x in content]
	# content = [content[i].split() for i in range(len(content))]
	# content = np.array(content)
	# content = np.reshape(content, [-1, ])\
	content = np.load(fname).item()
	return content['D']

def build_dataset(words):
	count = collections.Counter(words).most_common()
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary, reverse_dictionary

with tf.name_scope("drama") as scope:
	start_time = time.time()


	def elapsed(sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"


	# Target log path
	logs_path = './tf_logs_lm.log'
	writer = tf.summary.FileWriter(logs_path)

	# Text file containing words for training
	training_file = '/data/anussank/shavak/dict_data_wonan.npy'

	training_data = read_data(training_file)
	training_data = getSentences(training_data)
	print("Loaded training data...")

	embed_matrix = PretrainedEmbeddings("/data/anussank/shavak/GoogleNews-vectors-negative300.bin")
	# loaded_data = load_training_data(args.data_dir, args.data_set)
	# print loaded_data

	word_to_id, id_to_word = build_vocabulary(training_data, min_count=1)
	embed_matrix = embed_matrix.load_embedding_matrix(word_to_id)
	vocab_size = len(word_to_id)

	# Parameters
	learning_rate = 0.001
	training_iters = 50000
	display_step = 1000
	n_input = 3

	# number of units in RNN cell
	n_hidden = 512

	# tf Graph input
	x = tf.placeholder("float", [None, n_input, 1])
	y = tf.placeholder("float", [None, vocab_size])

	# RNN output node weights and biases
	weights = {
		'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
	}
	biases = {
		'out': tf.Variable(tf.random_normal([vocab_size]))
	}


	pred = RNN(x, weights, biases)

	# Loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Model evaluation
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as session:
		session.run(init)
		step = 0
		offset = random.randint(0, n_input + 1)
		end_offset = n_input + 1
		acc_total = 0
		loss_total = 0

		writer.add_graph(session.graph)

		while step < training_iters:
			# Generate a minibatch. Add some randomness on selection process.
			if offset > (len(training_data) - end_offset):
				offset = random.randint(0, n_input + 1)

			symbols_in_keys = [[word_to_id[str(training_data[i])]] for i in range(offset, offset + n_input)]
			symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

			symbols_out_onehot = np.zeros([vocab_size], dtype=float)
			symbols_out_onehot[word_to_id[str(training_data[offset + n_input])]] = 1.0
			symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

			_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
			                                        feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
			loss_total += loss
			acc_total += acc
			if (step + 1) % display_step == 0:
				print("Iter= " + str(step + 1) + ", Average Loss= " + \
				      "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
				      "{:.2f}%".format(100 * acc_total / display_step))
				acc_total = 0
				loss_total = 0
				symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
				symbols_out = training_data[offset + n_input]
				symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
				print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
			step += 1
			offset += (n_input + 1)
		# print("Optimization Finished!")
		# print("Elapsed time: ", elapsed(time.time() - start_time))
		# print("Run on command line.")
		# print("\ttensorboard --logdir=%s" % (logs_path))
		# print("Point your web browser to: http://localhost:6006/")
		# while True:
		# 	prompt = "%s words: " % n_input
		# 	sentence = input(prompt)
		# 	sentence = sentence.strip()
		# 	words = sentence.split(' ')
		# 	if len(words) != n_input:
		# 		continue
		# 	try:
		# 		symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
		# 		for i in range(32):
		# 			keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
		# 			onehot_pred = session.run(pred, feed_dict={x: keys})
		# 			onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
		# 			sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
		# 			symbols_in_keys = symbols_in_keys[1:]
		# 			symbols_in_keys.append(onehot_pred_index)
		# 		print(sentence)
		# 	except:
		# 		print("Word not in dictionary")
