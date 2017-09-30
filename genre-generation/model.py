import tensorflow as tf
from Utils import ops

class GAN:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''
	def __init__(self, options):
		self.options = options

		# self.g_bn0 = ops.batch_norm(name='g_bn0')
		# self.g_bn1 = ops.batch_norm(name='g_bn1')
		# self.g_bn2 = ops.batch_norm(name='g_bn2')
		# self.g_bn3 = ops.batch_norm(name='g_bn3')
		#
		# self.d_bn1 = ops.batch_norm(name='d_bn1')
		# self.d_bn2 = ops.batch_norm(name='d_bn2')
		# self.d_bn3 = ops.batch_norm(name='d_bn3')
		# self.d_bn4 = ops.batch_norm(name='d_bn4')


	def build_model(self):
		# img_size = self.options['image_size']
		# t_genre_sent = tf.placeholder('float32', [self.options['batch_size'],self.options['seq_len'], self.options['embed_len'], 1 ], name = 'real_genre')
		# t_not_genre_sent = tf.placeholder('float32', [self.options['batch_size'],self.options['seq_len'], self.options['embed_len'], 1 ], name = 'wrong_genre')
		# t_genre = tf.placeholder('float32', [self.options['batch_size'], self.options['embed_len']], name = 'genre')
		# t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		t_genre_sent = tf.placeholder('int64', [self.options['batch_size'],self.options['seq_len']], name = 'real_genre')
		t_not_genre_sent = tf.placeholder('int64', [self.options['batch_size'],self.options['seq_len']], name = 'wrong_genre')
		t_genre = tf.placeholder('int64', [self.options['batch_size']], name = 'genre')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		# t_labels = tf.placeholder('float32', [self.options['batch_size'], 5])


		with tf.variable_scope('embeddings_encoder', reuse = True):
			embeddings = tf.get_variable('embeddings')
			genre_embeddings = tf.get_variable('genre_embeddings')
			real_genre = tf.nn.embedding_lookup(embeddings, t_genre_sent)
			real_genre = tf.expand_dims(real_genre, -1)
			wrong_genre = tf.nn.embedding_lookup(embeddings, t_not_genre_sent)
			wrong_genre = tf.expand_dims(wrong_genre, -1)
			genre_list = tf.nn.embedding_lookup(genre_embeddings, t_genre)
			# genre_list = t_genre


		generated_sentence = self.generator(t_z, genre_list)
		# GENRE
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			disc_genre_sent, disc_genre_sent_logits   = self.discriminator(real_genre, genre_list)
			disc_not_genre_sent, disc_not_genre_logits   = self.discriminator(wrong_genre, genre_list, reuse = True)
			disc_gen_sent, disc_gen_sent_logits   = self.discriminator(generated_sentence, genre_list, reuse = True)

		# zeros = tf.random_normal(disc_gen_sent.shape(), 0.15, 0.15)
		# ones = tf.random_normal(disc_gen_sent.shape(), 0.95, 0.25)
		# noisy_zeros =
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_gen_sent_logits, labels=tf.ones_like(disc_gen_sent)))
		# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_gen_sent_logits, labels=zeros))

		# d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image)))
		# d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image)))
		# d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image)))
		# GENRE
		d_loss1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_genre_sent_logits, labels=tf.ones_like(disc_genre_sent_logits)))
		d_loss2 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_not_genre_logits, labels=tf.zeros_like(disc_not_genre_logits)))
		d_loss3 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_gen_sent_logits, labels=tf.zeros_like(disc_gen_sent_logits)))

		d_loss = d_loss1 + d_loss2 + d_loss3
		# d_loss =  d_loss1 + d_loss3

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		input_tensors = {
			't_genre_sent' : t_genre_sent,
			't_not_genre_sent' : t_not_genre_sent,
			't_genre' : t_genre,
			't_z' : t_z
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : generated_sentence
		}
		#GENRE
		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'disc_genre_sent_logits' : disc_genre_sent_logits,
			'disc_not_genre_sent' : disc_not_genre_sent,
			'disc_gen_sent_logits' : disc_gen_sent_logits
		}
		
		return input_tensors, variables, loss, outputs, checks


	# #Not used for training
	# def build_generator(self):
	# 	img_size = self.options['image_size']
	# 	t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
	# 	t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
	# 	fake_image = self.sampler(t_z, t_real_caption)
	#
	# 	input_tensors = {
	# 		't_real_caption' : t_real_caption,
	# 		't_z' : t_z
	# 	}
	#
	# 	outputs = {
	# 		'generator' : fake_image
	# 	}
	#
	# 	return input_tensors, outputs
	def build_generator(self):
		# img_size = self.options['image_size']
		t_genre = tf.placeholder('int64', [self.options['batch_size']],
		                                name='real_sent_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		generated_sent = self.sampler(t_z, t_genre)

		input_tensors = {
			't_genre': t_genre,
			't_z': t_z
		}

		outputs = {
			'generator': generated_sent
		}

		return input_tensors, outputs

	# Sample Images for a text embedding
	# def sampler(self, t_z, t_text_embedding):
	# 	tf.get_variable_scope().reuse_variables()
	#
	# 	s = self.options['image_size']
	# 	s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
	#
	# 	reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
	# 	z_concat = tf.concat(1, [t_z, reduced_text_embedding])
	# 	z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
	# 	h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
	# 	h0 = tf.nn.relu(self.g_bn0(h0, train = False))
	#
	# 	h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
	# 	h1 = tf.nn.relu(self.g_bn1(h1, train = False))
	#
	# 	h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
	# 	h2 = tf.nn.relu(self.g_bn2(h2, train = False))
	#
	# 	h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
	# 	h3 = tf.nn.relu(self.g_bn3(h3, train = False))
	#
	# 	h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
	#
	# 	return (tf.tanh(h4)/2. + 0.5)

	def sampler(self, t_z, t_text_embedding):

		tf.get_variable_scope().reuse_variables()

		with tf.variable_scope('embeddings_encoder', reuse=True) as scope:
			embeddings = tf.get_variable('genre_embeddings')
			embedList = tf.nn.embedding_lookup(embeddings, t_text_embedding)
			t_text_embedding = embedList

		reduced_text_embedding = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
		z_concat = tf.concat([t_z, reduced_text_embedding], 1)
		# z_concat = t_z
		z_ = ops.linear(z_concat, self.options['t_dim'], 'g_embedding_noise')
		cell = tf.contrib.rnn.GRUCell(num_units=self.options['num_lstm_units'])
		output, state = ops.rnn(z_, seq_len=self.options['seq_len'], cell=cell, name='g_rnn',
		                        batch_size=self.options['batch_size'])
		# print(output)
		output = tf.expand_dims(output, axis=-1)
		return output

	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		# s = self.options['image_size']
		# s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		# GENRE
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding], 1)
		# z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		# h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		# h0 = tf.nn.relu(self.g_bn0(h0))
		#
		# h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		# h1 = tf.nn.relu(self.g_bn1(h1))
		#
		# h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		# h2 = tf.nn.relu(self.g_bn2(h2))
		#
		# h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		# h3 = tf.nn.relu(self.g_bn3(h3))
		#
		# h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		z_ = ops.linear(z_concat, self.options['t_dim'], 'g_embedding_noise')
		cell = tf.contrib.rnn.GRUCell(num_units = self.options['num_lstm_units'])
		output, state = ops.rnn(z_, seq_len=self.options['seq_len'], cell = cell, name = 'g_rnn', batch_size=self.options['batch_size'])
		# print(output)
		output = tf.expand_dims(output, axis=-1)
		return output

		# return (tf.tanh(h4)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

	# tells whether given sentence belongs to given genre or not
	def discriminator(self, t_text_embedding, t_genre_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		# h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		# h1 = ops.lrelu( self.d_bn1(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))) #16
		# h2 = ops.lrelu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))) #8
		# h3 = ops.lrelu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))) #4
		#
		# ADD TEXT EMBEDDING TO THE NETWORK


		# reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
		# reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		# reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		# tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		#
		# h3_concat = tf.concat( 3, [h3, tiled_embeddings], name='h3_concat')
		# h3_new = ops.lrelu( self.d_bn4(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		#
		# h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')

		# return tf.nn.sigmoid(h4), h4


		numFilters = 128
		filter_sizes = [3, 4, 5]
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("d_conv-maxpool-%s" % filter_size):
				# Convolution Layer
				# print (t_text_embedding)
				filter_shape = [filter_size, self.options['t_dim'], 1, numFilters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[numFilters]), name="b")
				conv = tf.nn.conv2d(
					t_text_embedding,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				conv = tf.layers.dropout(conv, rate=0.6, training=True)
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, self.options['seq_len'] - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = numFilters * len(filter_sizes)

		h_pool = tf.concat(pooled_outputs, 3)
		h_pool_flat = ops.linear(tf.reshape(h_pool, [-1, num_filters_total]), 1, 'scores_flatten')
		# print h_pool_flat
		#GENRE
		h_pool_flat = tf.concat([h_pool_flat, t_genre_embedding], 1)
		# with tf.variable_scope(tf.get_variable_scope()) as scope:
		scores = ops.linear(h_pool_flat, 1, 'scores')

			# self.predictions = tf.argmax(self.scores, 1, name="predictions")
		return tf.nn.sigmoid(scores), scores
