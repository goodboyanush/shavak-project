import setGPU
import tensorflow as tf
import model
import argparse
from embed_help_oov import *
import numpy as np
from collections import defaultdict
from train import genre2ID
from collections import Counter


def generateGenreEmbed(genreList, genreID):
	embedList = []
	for k in genreList:
		# embed = generate_embedding(word_to_id, embed_matrix, [k], tokenize=False)
		embed = genreID[k]
		embedList.append([k, int(embed)])

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

def getSentenceFromId(ids, id_to_word):
	sent = []
	for i in xrange(0, len(ids)):
		sent.append("".join(id_to_word[ids[i][0]]))
	sent = " ".join(sent)
	return sent

def getBagFromId(ids, id_to_word):
	sent = []
	for i in xrange(0, len(ids)):
		sent.append("".join(id_to_word[ids[i][0]]))
	# sent = " ".join(sent)
	return sent


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--z_dim', type=int, default=100,
	                    help='Noise Dimension')

	parser.add_argument('--t_dim', type=int, default=300,
	                    help='Text feature dimension')

	parser.add_argument('--data_dir', type=str, default="/data/anussank/shavak/newsdata/sample_data.npy",
	                    help='Data Directory')

	parser.add_argument('--embed_path', type=str, default="/data/anussank/shavak/GoogleNews-vectors-negative300.bin",
	                    help='Embeddings Path')

	parser.add_argument('--seq_len', type=int, default=20,
	                    help='Sentence Length')

	parser.add_argument('--model_path', type=str,
	                    default='/data/anussank/shavak/GAN/news_3/-164',
	                    help='Trained Model Path')

	parser.add_argument('--n_sent', type=int, default=20,
	                    help='Number of Sentences per Genre')

	parser.add_argument('--num_lstm_units', type=int, default=300,
	                    help='Number of LSTM Units')

	args = parser.parse_args()
	model_options = {
		'z_dim': args.z_dim,
		't_dim': args.t_dim,
		'batch_size': args.n_sent,
		'seq_len': args.seq_len,
		'embed_len': 300,
		'vocab_size': 0,
		'num_lstm_units': 300,
	}


	loaded_data = np.load(args.data_dir).item()
	word_to_id, id_to_word = build_vocabulary(getSentences(loaded_data), loaded_data.keys(), min_count=1)
	model_options['vocab_size'] = len(word_to_id)
	with tf.variable_scope("embeddings_encoder"):
		# embed_matrix = tf.get_variable("embeddings")
		embed_matrix = tf.get_variable('embeddings',
		                            initializer=tf.truncated_normal([len(word_to_id), args.t_dim], mean=0.0,
		                                                            stddev=0.01), trainable=False)
		genre_embedding = tf.get_variable('genre_embeddings',
		                                  initializer=tf.truncated_normal([len(loaded_data.keys()), args.t_dim],
		                                                                  mean=0.0,
		                                                                  stddev=0.01), trainable=False)
	gan = model.GAN(model_options)
	_, _, _, _, _ = gan.build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)

	input_tensors, outputs = gan.build_generator()

	# embed_matrix = embed_matrix.load_embedding_matrix(word_to_id)
	with tf.variable_scope("embeddings_encoder", reuse=True):
		embed_matrix = tf.get_variable("embeddings")
		embed_matrix = sess.run(embed_matrix)
	normed_embedding = tf.nn.l2_normalize(embed_matrix, dim=1)
	genreID = genre2ID(loaded_data)
	embedded_genre = generateGenreEmbed(loaded_data.keys(), genreID)

	caption_image_dic = defaultdict(list)
	for k, genre_embed in (embedded_genre):
		genre_sent = []
		z_noise = np.random.uniform(-1, 1, [args.n_sent, args.z_dim])
		genre_embed = [genre_embed] * args.n_sent
		[gen_sent] = sess.run([outputs['generator']],
		                      feed_dict={
			                      input_tensors['t_genre']: genre_embed,
			                      input_tensors['t_z']: z_noise,
		                      })
		# print (gen_sent[0])
		genre_sent = [gen_sent[i] for i in range(0, args.n_sent)]
		caption_image_dic[k].extend(genre_sent)
		print "Generated", k

	with open("/home/anussank/shavak/genre-generation-argmax/news_3/freq_out_164.txt", 'w+') as f:
		for genres in caption_image_dic.keys():
			wordcount = Counter()
			for i in xrange(0, len(caption_image_dic[genres])):
				# print caption_image_dic[genres][i]
				# sentence = getSentenceFromEmbed(normed_embedding, caption_image_dic[genres][i], id_to_word, sess)
				# sentence = getSentenceFromId(caption_image_dic[genres][i], id_to_word)
				bag = getBagFromId(caption_image_dic[genres][i], id_to_word)
				wordcount.update(bag)
				# f.write(genres + "\t" + sentence + '\n')
				# f.write(genres + " " + sentence + '\n')
			f.write("_______________________________________________________________________________\n")
			f.write(genres + '\n')
			f.write("_______________________________________________________________________________\n")
			# print (wordcount.most_common(20)[0] )
			for c in wordcount.most_common(20):
				f.write(c[0].decode('ascii', 'ignore') + ' ' + str(c[1]) + '\n')
			# f.write(wordcount.most_common())
		f.close()


if __name__ == '__main__':
	main()
