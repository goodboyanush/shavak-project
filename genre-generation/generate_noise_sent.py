import tensorflow as tf
import model
import argparse
# import pickle
# from os.path import join
from embed_help import *
from train import generate_embedding
# import h5py
# from Utils import image_processing
# import scipy.misc
# import random
# import json
# import os
import numpy as np
from collections import defaultdict

def generateGenreEmbed(genreList, word_to_id, embed_matrix):
	embedList = []
	for k in genreList:
		embed = generate_embedding(word_to_id, embed_matrix, [k], tokenize=False)
		embedList.append([k, embed[0]])

	return embedList

def getSentenceFromEmbed(normedEmbed, sentEmbed, id_to_word, sess):
	normed_array = tf.nn.l2_normalize(sentEmbed, dim=1)
	cosine_similarity = tf.matmul(normed_array, tf.transpose(normedEmbed, [1, 0]))
	closest_words = tf.argmax(cosine_similarity, 1)
	closest_words = sess.run(closest_words)
	sent = []
	for words in closest_words:
		# print id_to_word[words]
		sent.append("".join(id_to_word[words]))

	sent = " ".join(sent)

	return sent


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise Dimension')

	parser.add_argument('--t_dim', type=int, default=300,
					   help='Text feature dimension')

	parser.add_argument('--data_dir', type=str, default="/data/anussank/shavak/sanity.npy",
	                    help='Data Directory')

	parser.add_argument('--embed_path', type=str, default="/data/anussank/shavak/GoogleNews-vectors-negative300.bin",
	                    help='Embeddings Path')

	# parser.add_argument('--image_size', type=int, default=64,
	# 				   help='Image Size')
	#
	# parser.add_argument('--gf_dim', type=int, default=64,
	# 				   help='Number of conv in the first layer gen.')
	#
	# parser.add_argument('--df_dim', type=int, default=64,
	# 				   help='Number of conv in the first layer discr.')
	#
	# parser.add_argument('--gfc_dim', type=int, default=1024,
	# 				   help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--seq_len', type=int, default=20,
					   help='Sentence Length')

	parser.add_argument('--model_path', type=str, default='/data/anussank/shavak/Model_sanity/model_after_epoch_1495.ckpt',
                       help='Trained Model Path')

	parser.add_argument('--n_sent', type=int, default=5,
                       help='Number of Sentences per Caption')

	parser.add_argument('--num_lstm_units', type=int, default=300,
						help='Number of LSTM Units')


	# parser.add_argument('--caption_thought_vectors', type=str, default='Data/sample_caption_vectors.hdf5',
     #                   help='Caption Thought Vector File')

	
	args = parser.parse_args()
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.n_sent,
		'seq_len': args.seq_len,
		'embed_len': 300,
		'num_lstm_units': 300,
		# 'image_size' : args.image_size,
		# 'gf_dim' : args.gf_dim,
		# 'df_dim' : args.df_dim,
		# 'gfc_dim' : args.gfc_dim,
		# 'caption_vector_length' : args.caption_vector_length
	}

	gan = model.GAN(model_options)
	_, _, _, _, _ = gan.build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)
	
	input_tensors, outputs = gan.build_generator()

	# h = h5py.File( args.caption_thought_vectors )
	# caption_vectors = np.array(h['vectors'])
	embed_matrix = PretrainedEmbeddings(args.embed_path)
	# loaded_data = load_training_data(args.data_dir, args.data_set)
	loaded_data = np.load(args.data_dir).item()
	# print loaded_data
	word_to_id, id_to_word = build_vocabulary(getSentences(loaded_data), loaded_data.keys(), min_count=1)

	embed_matrix = embed_matrix.load_embedding_matrix(word_to_id)
	normed_embedding = tf.nn.l2_normalize(embed_matrix, dim=1)
	embedded_genre = generateGenreEmbed(loaded_data.keys(), word_to_id, embed_matrix)


	caption_image_dic = defaultdict(list)
	for k, genre_embed in (embedded_genre):

		genre_sent = []
		z_noise = np.random.uniform(-1, 1, [args.n_sent, args.z_dim])
		genre_embed = [ genre_embed ] * args.n_sent
		
		[ gen_sent ] = sess.run( [ outputs['generator'] ],
			feed_dict = {
				input_tensors['t_genre'] : genre_embed,
				input_tensors['t_z'] : z_noise,
			} )
		
		genre_sent = [np.reshape(gen_sent[i,:,:,:], (args.seq_len, args.t_dim)) for i in range(0, args.n_sent)]
		caption_image_dic[ k ].extend(genre_sent)
		print "Generated", k

	# for f in os.listdir( join(args.data_dir, 'val_samples')):
	# 	if os.path.isfile(f):
	# 		os.unlink(join(args.data_dir, 'val_samples/' + f))
	with open("/home/anussank/shavak/text-to-image-master/out.txt", 'w+') as f:
		for genres in caption_image_dic.keys():
			for i in xrange(0, len(caption_image_dic[genres])):
				# print caption_image_dic[genres][i]
				sentence = getSentenceFromEmbed(normed_embedding, caption_image_dic[genres][i], id_to_word, sess)
				f.write(genres + "\t" + sentence + '\n')

		f.close()

	# for cn in range(0, len(caption_vectors)):
	# 	genre_image = []
	# 	for i, im in enumerate( caption_image_dic[ cn ] ):
	# 		# im_name = "caption_{}_{}.jpg".format(cn, i)
	# 		# scipy.misc.imsave( join(args.data_dir, 'val_samples/{}'.format(im_name)) , im)
	# 		genre_image.append( im )
	# 		genre_image.append( np.zeros((64, 5, 3)) )
	# 	combined_image = np.concatenate( genre_image[0:-1], axis = 1 )
		# scipy.misc.imsave( join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn)) , combined_image)


if __name__ == '__main__':
	main()
