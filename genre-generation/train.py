import setGPU
import tensorflow as tf
import numpy as np
import model
import argparse
from os.path import join
import scipy.misc
import os
import shutil
from nltk.tokenize import RegexpTokenizer

from embed_help_oov import build_vocabulary, getSentences


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default=300,
					   help='Text feature dimension/ Embedding Dimension')

	parser.add_argument('--batch_size', type=int, default=64,
					   help='Batch ize')

	# parser.add_argument('--max_sent_len', type=int, default=10,
	# 				   help='Maximum length of a sentence')

	parser.add_argument('--data_dir', type=str, default="/data/anussank/shavak/split_data.npy",
					   help='Data Directory')

	parser.add_argument('--embed_path', type=str, default="/data/anussank/shavak/GoogleNews-vectors-negative300.bin",
	                    help='Embeddings Path')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
					   help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
					   help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=200,
					   help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=30,
					   help='Save Model/Samples every x iterations over batches')

	parser.add_argument('--log_dir', type=str, default="/data/anussank/shavak/GAN/seqlen_10/",
                       help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--num_lstm_units', type=int, default=300,
						help='Number of LSTM Units')

	parser.add_argument('--seq_len', type=int, default=10,
	                    help='Maximum Length of Sentence')

	args = parser.parse_args()
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'num_lstm_units': args.num_lstm_units,
		'seq_len': args.seq_len,
		'embed_len': 300
	}


	with tf.variable_scope(tf.get_variable_scope()):

		loaded_data = load_training_data(args.data_dir)
		word_to_id, id_to_word = build_vocabulary(getSentences(loaded_data), loaded_data.keys(), min_count=1)
		with tf.variable_scope('embeddings_encoder') as scope:
			embedding = tf.get_variable('embeddings',
			                            initializer=tf.truncated_normal([len(word_to_id), args.t_dim], mean=0.0,
			                                                            stddev=0.01), trainable=True)
			genre_embedding = tf.get_variable('genre_embeddings',
			                            initializer=tf.truncated_normal([len(loaded_data.keys()), args.t_dim], mean=0.0,
			                                                            stddev=0.01), trainable=True)

			ass_1 = embedding.assign(np.load("/data/anussank/shavak/GAN/indiv_GRU_OOV/Model_embedding:0.npy"))

		gan = model.GAN(model_options)
		input_tensors, variables, loss, outputs, checks = gan.build_model()

		# print (loss, variables)

		d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
		g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

		with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
			ass_2 = tf.get_variable("g_rnn/gru_cell/gates/kernel").assign(np.load(
				"/data/anussank/shavak/GAN/indiv_GRU_OOV/Model_RNN_multi_rnn_cell_cell_0_gru_cell_gates_kernel:0.npy"))
			ass_3 = tf.get_variable("g_rnn/gru_cell/gates/bias").assign(np.load(
				"/data/anussank/shavak/GAN/indiv_GRU_OOV/Model_RNN_multi_rnn_cell_cell_0_gru_cell_gates_bias:0.npy"))
			ass_4 = tf.get_variable("g_rnn/gru_cell/candidate/kernel").assign(np.load(
				"/data/anussank/shavak/GAN/indiv_GRU_OOV/Model_RNN_multi_rnn_cell_cell_0_gru_cell_candidate_kernel:0.npy"))
			ass_5 = tf.get_variable("g_rnn/gru_cell/candidate/bias").assign(np.load(
				"/data/anussank/shavak/GAN/indiv_GRU_OOV/Model_RNN_multi_rnn_cell_cell_0_gru_cell_candidate_bias:0.npy"))

		sv = tf.train.Supervisor(logdir=args.log_dir, saver=tf.train.Saver(max_to_keep=2000))
		protoconfig = tf.ConfigProto(allow_soft_placement=True)
		protoconfig.gpu_options.allow_growth = True
		with sv.managed_session(config=protoconfig) as sess:

			# embed_matrix = PretrainedEmbeddings(args.embed_path)
			# loaded_data = load_training_data(args .data_dir, args.data_set)
			# print loaded_data
			# embed_matrix = embed_matrix.load_embedding_matrix(word_to_id)
			# sess.run([ass_1, ass_2, ass_3, ass_4, ass_5])
			# sess.run(ass_1)
			# for var in tf.global_variables():
			# 	print var
			start = 8
			for i in xrange(start, args.epochs):
				batch_no = 0
				last_index = 0
				ended = False
				print ("__________________________________________________________________________________________________________")
				print i
				print("Saving model to %s." % args.log_dir)
				sv.saver.save(sess, args.log_dir, global_step=i)
				print ("__________________________________________________________________________________________________________")
				while not ended:
					real_genre, wrong_genre, genre_list, z_noise, last_index, ended = get_training_batch(sess, args.batch_size,
						args.t_dim, args.z_dim, args.seq_len, last_index, embedding, word_to_id, loaded_data)

					# DISCR UPDATE
					check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
					# check_ts = [checks['d_loss1'], checks['d_loss3']]

					# _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
					# 	feed_dict = {
					# 		input_tensors['t_genre_sent'] : real_genre,
					# 		input_tensors['t_not_genre_sent'] : wrong_genre,
					# 		input_tensors['t_genre'] : genre_list,
					# 		input_tensors['t_z'] : z_noise,
					# 	})
					#
					for k in xrange(0, 15):
						_, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
							feed_dict = {
								input_tensors['t_genre_sent'] : real_genre,
								input_tensors['t_not_genre_sent'] : wrong_genre,
								input_tensors['t_genre'] : genre_list,
								input_tensors['t_z'] : z_noise,
							})

					# GEN UPDATE
					for k in xrange(0, 1):
						_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
							feed_dict = {
								input_tensors['t_genre_sent'] : real_genre,
								input_tensors['t_not_genre_sent'] : wrong_genre,
								input_tensors['t_genre'] : genre_list,
								input_tensors['t_z'] : z_noise,
							})

					if (batch_no+1)%5 == 0:
						print "d1", d1
						print "d2", d2
						print "d3", d3
						print "D", d_loss
						print "LOSSES", d_loss, g_loss, batch_no, i
					batch_no += 1
				start = 0
					# if (batch_no % args.save_every) == 0:
					# 	print "Saving Model"
					# 	save_path = saver.save(sess, "/data/anussank/shavak/Models/latest_model_temp.ckpt")
				# if (i+1)%5 == 0:
				# if True:
					# print("Saving model to %s." % args.log_dir)
					# sv.saver.save(sess, args.log_dir, global_step=i)
					# save_path = sv.saver.save(sess, args.log_dir+"model_after_epoch_{}.ckpt".format(i))

def load_training_data(data_dir):
	data = np.load(data_dir).item()
	return data

def save_for_vis(data_dir, real_images, generated_images, image_files):
	
	shutil.rmtree( join(data_dir, 'samples') )
	os.makedirs( join(data_dir, 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)

def generate_embedding(word_to_id, embed_matrix, sentence, max_sent_len = 15, tokenize = True):
	# sentence = sentence[1]
	tokenizer = RegexpTokenizer('\w+')
	embed = []
	if tokenize:
		tokens = tokenizer.tokenize(sentence)
		try:
			tokens = map(str.lower, tokens)
		except TypeError:
			try:
				tokens = map(unicode.lower, tokens)
			except TypeError:
				print tokens
				raise
	else:
		tokens = sentence
	if(len(tokens)>max_sent_len):
		tokens = tokens[:max_sent_len]
	for word in tokens:
		try:
			embed.append(word_to_id[word])
		except KeyError:
			try:
				embed.append(word_to_id["<OOV>"])
			except KeyError:
				print (tokens)
				print (sentence)
				raise
	return embed

def genre2ID(loaded_data):
	temp = {}
	i = 0
	for k in loaded_data.keys():
		temp[k] = i
		i += 1
	return temp

def getGenreSentPairs(loaded_data):
	pairs = []
	for k in loaded_data.keys():
		for sent in loaded_data[k]:
			pairs.append([k, sent])
	return pairs

def getWithoutGenre(loaded_data, genre):
	filtered = {}
	for k in loaded_data.keys():
		if (k!=genre):
			filtered[k] = loaded_data[k]
	return filtered
# def get_training_batch(batch_no, batch_size, image_size, z_dim,
# 	caption_vector_length, split, data_dir, data_set, loaded_data = None):
def get_training_batch(sess, batch_size, embed_size, z_dim,
	                       max_sent_len, last_index, embed_matrix, word_to_id, loaded_data=None):

	# real_genre = np.zeros((batch_size, max_sent_len, embed_size, 1))
	# wrong_genre = np.zeros((batch_size, max_sent_len, embed_size, 1))

	real_genre = np.zeros((batch_size, max_sent_len), dtype=np.int32)
	wrong_genre = np.zeros((batch_size, max_sent_len), dtype=np.int32)

	# keys = loaded_data.keys()
	size = 0
	ctr = 0
	genre_list = []
	ended = False
	loaded_data_ = getGenreSentPairs(loaded_data)
	genreID = genre2ID(loaded_data)
	# print len(loaded_data_)
	# filtered_data = getGenreSentPairs(getWithoutGenre(loaded_data, loaded_data_[last_index][0]))

	while(size<batch_size):
		# embed_sent = generate_embedding(word_to_id, embed_matrix, loaded_data[last_index], max_sent_len)
		# print loaded_data_[last_index]
		# print len(loaded_data_), size, batch_size

		embed = generate_embedding(word_to_id, embed_matrix, loaded_data_[last_index][1], max_sent_len)
		if(len(embed) == 0):
			last_index+=1
			continue
		# real_genre[size, :len(embed), :, 0] = embed
		# print embed
		real_genre[size, :len(embed)] = embed
		# genre_in_real.add(loaded_data_[last_index][0])
		# genre_list.append(generate_embedding(word_to_id, embed_matrix, [loaded_data_[last_index][0]], max_sent_len, tokenize=False)[0])
		genre_list.append(genreID[loaded_data_[last_index][0]])

		index = np.random.randint(low=0, high=len(loaded_data_))
		while((loaded_data_[index][0] == loaded_data_[last_index][0]) or (len(embed)==0)):
			index = np.random.randint(low=0, high=len(loaded_data_))
			embed = generate_embedding(word_to_id, embed_matrix, loaded_data_[index][1], max_sent_len)
		try:
			# wrong_genre[size, :len(embed), :, 0] = embed
			wrong_genre[size, :len(embed)] = embed
		except ValueError:
			print loaded_data_[index][1]
			raise
		size += 1
		last_index+=1

		if(last_index==len(loaded_data_)):
			# print len(loaded_data_), size, batch_size
			last_index = 0
			ended = True

	#TODO>> As of Now, wrong images are just shuffled real images.
	# first_sent = real_genre[0,:,:,:]
	# for i in range(0, batch_size):
	# 	if i < batch_size - 1:
	# 		wrong_genre[i,:,:,:] = real_genre[i+1,:,:,:]
	# 	else:
	# 		wrong_genre[i,:,:,:] = first_sent

	# z_noise = np.random.normal(0, 1, [batch_size, z_dim])
	z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
	# genre_list = np.vstack([np.expand_dims(x, 0) for x in genre_list])
	# genre_list = np.vstack(genre_list)
	# print len(genre_list), genre_list
	return real_genre.astype(np.int32), wrong_genre.astype(np.int32), genre_list, z_noise, last_index, ended

if __name__ == '__main__':
	main()
