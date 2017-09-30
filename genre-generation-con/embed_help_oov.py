# import gensim
import numpy as np
import time
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet

from collections import Counter

__author__ = "Vikas Raykar,Anirban Laha"
__email__ = "viraykar@in.ibm.com"

# __all__ = ["PretrainedEmbeddings"]

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
		# for token in tokens:
		# 	token.encode('ascii')
		try:
			tokens = map(str.lower, tokens)
		except TypeError:
			# try:
			tokens = map(unicode.lower, tokens)

		for i in xrange(0, len(tokens)):
			if not wordnet.synsets(tokens[i]):
				tokens[i] = "<OOV>"
		# except TypeError:
		# 	print tokens
		# 	raise
		wordcount.update(tokens)
		# print tokens

	# for k in genres:
	# 	# map(str.lower, k)
	# 	print k
	# 	wordcount.update([k])

	print('vocabulary size = %d' % (len(wordcount)))
	# filtering
	count_pairs = wordcount.most_common()
	count_pairs = [c for c in count_pairs if c[1] >= 1]
	words, _ = list(zip (*count_pairs))
	word_to_id = dict(zip(words, range(1, len(words) + 1)))

	word_to_id['<EOS>'] = 0
	# word_to_id['<OOV>'] = 1
	print('vocabulary size = %d (after filtering with min_count =  %d)' % (len(word_to_id), len(words)))
	id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
	return word_to_id, id_to_word


# class PretrainedEmbeddings():
# 	""" load the pre-trained embeddings
# 	"""
#
# 	def __init__(self, filename):
# 		""" load the pre-trained embeddings
#
# 		:params:
# 			filename : string
# 				full path to the .bin file containing the pre-trained word vectors
# 				GoogleNews-vectors-negative300.bin
# 				can be downloaded from https://code.google.com/archive/p/word2vec/
# 		"""
#
# 		_, file_extension = os.path.splitext(filename)
# 		if file_extension == '.bin':
# 			is_binary = True
# 		else:
# 			is_binary = False
#
# 		print('Loading the pre-trained embeddings from [%s].' % (filename))
# 		start = time.time()
# 		# Following works in cpu server
# 		self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=is_binary)
# 		# Following worls in ccc
# 		# self.model = gensim.models.KeyedVectors.load_word2vec_format(filename,binary=is_binary)
# 		time_taken = time.time() - start
# 		print('Takes %2.2f secs !' % (time_taken))
#
# 		self._embedding_size = 300
#
# 	def load_embedding_matrix(self, word_to_id):
# 		""" extract the embedding matrix for a given vocabulary
#
# 		words not in word2vec are intialized randonaly to uniform(-a,a), where a is chosen such that the
# 		unknown words have the same variance as words already in word_to_id
#
# 		:params:
# 			word_to_id: dict
# 				dict mapping a word to its id, e.g., word_to_id['the'] = 3
#
# 		:returns:
# 			embedding_matrix: np.array, float32 - [vocab_size, embedding_size]
# 				the embedding matrix
# 		"""
# 		vocab_size = len(word_to_id)
#
# 		embedding_matrix = 100.0 * np.ones((vocab_size, self._embedding_size)).astype('float32')
#
# 		count = 0
# 		for word in word_to_id:
# 			if word in self.model:
# 				embedding_matrix[word_to_id[word], :] = self.model[word]
# 				count += 1
# 		print('Found pre-trained word2vec embeddings for %d/%d words.' % (count, vocab_size))
#
# 		init_OOV = np.sqrt(3.0 * np.var(embedding_matrix[embedding_matrix != 100.0]))
# 		embed_OOV = np.random.uniform(-init_OOV, init_OOV, (1, self._embedding_size))
# 		for word in word_to_id:
# 			if word not in self.model:
# 				embedding_matrix[word_to_id[word], :] = np.random.uniform(-init_OOV, init_OOV,
# 				                                                          (1, self._embedding_size))
#
# 		return embedding_matrix
#
# 	@property
# 	def embedding_size(self):
# 		return self._embedding_size
