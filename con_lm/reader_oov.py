# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter
from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import wordnet, stopwords, words
import enchant
import collections
import math
import sys
import numpy as np
import tensorflow as tf


# def _read_words(filename):
# 	with tf.gfile.GFile(filename, "r") as f:
# 		if sys.version_info[0] >= 3:
# 			return f.read().replace("\n", "<eos>").split()
# 		else:
# 			return f.read().decode("utf-8").replace("\n", "<eos>").split()

d = enchant.Dict('en_US')

def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id


def getSentences(data):
	sent = []
	for k in data.keys():
		for sent_ in data[k]:
			sent.append(sent_)

	return sent


def build_vocab_(data):
	# sentences = getSentences(data)
	sentences = data
	wordcount = Counter()
	tokenizer = RegexpTokenizer(r'\w+')
	wordList = []
	# vocab_set = set(stopwords.words("english"))
	# vocab_set.update(words.words())
	for sentence in sentences:
		tokens = tokenizer.tokenize(sentence)
		# for token in tokens:
		# 	token.encode('ascii')
		try:
			tokens = map(str.lower, tokens)
		except TypeError:
			# try:
			tokens = map(unicode.lower, tokens)

		for i in xrange(0, len(tokens)):
			# if not wordnet.synsets(tokens[i].decode('utf-8', "ignore")):
			if not d.check(tokens[i].decode('utf-8', "ignore")):
				tokens[i] = "<OOV>"
		# except TypeError:
		# 	print tokens
		# 	raise
		wordList.extend(tokens)
		wordList.extend("<EOS>")
		wordcount.update(tokens)
	print('vocabulary size = %d' % (len(wordcount)))
	# filtering
	count_pairs = wordcount.most_common()
	count_pairs = [c for c in count_pairs if c[1] >= 1]
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(1, len(words) + 1)))
	print('vocabulary size = %d (after filtering with min_count =  %d)' % (len(word_to_id), 1))
	word_to_id['<EOS>'] = 0
	# word_to_id['<OOV>'] = 1
	id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
	return word_to_id, id_to_word, wordList

def _file_to_word_ids(data, word_to_id):
	# data = _read_words(filename)      
	return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
	"""Load PTB raw data from data directory "data_path".
  
	Reads PTB text files, converts strings to integer ids,
	and performs mini-batching of the inputs.
  
	The PTB dataset comes from Tomas Mikolov's webpage:
  
	http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  
	Args:
	  data_path: string path to the directory where simple-examples.tgz has
		been extracted.
  
	Returns:
	  tuple (train_data, valid_data, test_data, vocabulary)
	  where each of the data objects can be passed to PTBIterator.
	"""

	# train_path = os.path.join(data_path, "ptb.train.txt")
	# valid_path = os.path.join(data_path, "ptb.valid.txt")
	# test_path = os.path.join(data_path, "ptb.test.txt")

	data = np.load(data_path)
	# data = np.load(data_path).item()
	# f = open(data_path)
	# data = f.readlines()
	word_to_id, id_to_word, wordList = build_vocab_(data)
	# word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(wordList[int(len(wordList)*0.3):int(len(wordList)*1.0)], word_to_id)
	valid_data = _file_to_word_ids(wordList[int(len(wordList)*0.2):int(len(wordList)*0.3)], word_to_id)
	test_data = _file_to_word_ids(wordList[int(len(wordList)*0):int(len(wordList)*0.2)], word_to_id)
	vocabulary = len(word_to_id)
	return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
	"""Iterate on the raw PTB data.
  
	This chunks up raw_data into batches of examples and returns Tensors that
	are drawn from these batches.
  
	Args:
	  raw_data: one of the raw data outputs from ptb_raw_data.
	  batch_size: int, the batch size.
	  num_steps: int, the number of unrolls.
	  name: the name of this operation (optional).
  
	Returns:
	  A pair of Tensors, each shaped [batch_size, num_steps]. The second element
	  of the tuple is the same data time-shifted to the right by one.
  
	Raises:
	  tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
	"""
	with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

		data_len = tf.size(raw_data)
		batch_len = data_len // batch_size
		data = tf.reshape(raw_data[0: batch_size * batch_len],
		                  [batch_size, batch_len])

		epoch_size = (batch_len - 1) // num_steps
		assertion = tf.assert_positive(
			epoch_size,
			message="epoch_size == 0, decrease batch_size or num_steps")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name="epoch_size")

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
		x = tf.strided_slice(data, [0, i * num_steps],
		                     [batch_size, (i + 1) * num_steps])
		x.set_shape([batch_size, num_steps])
		y = tf.strided_slice(data, [0, i * num_steps + 1],
		                     [batch_size, (i + 1) * num_steps + 1])
		y.set_shape([batch_size, num_steps])
		return x, y
