import tensorflow as tf
import numpy as np

class TextProcessingMLP(object):
	def __init__(
		self,
		batch_size,
		layers_sizes,
	  max_sequence_length,
	  num_classes,
		word_vectors,
	  embedding_size):
			self.vocab_size = len(word_vectors)
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
			self.Y = tf.placeholder(tf.float32,
															[batch_size, num_classes],
															name="Y")

			self.X = tf.placeholder(tf.int32,
			                        [batch_size, max_sequence_length],
			                        name="X")
			with tf.name_scope("embedding"):
				self.embedded_reviews = tf.Variable(tf.zeros([batch_size, max_sequence_length, embedding_size]), dtype=tf.float32, name="emmbeded_reviews")
				self.embedded_reviews = tf.nn.embedding_lookup(word_vectors, self.X)
				self.embedded_reviews = tf.cast(self.embedded_reviews, tf.float32)
				print(self.embedded_reviews)
				# reducing dims , assembling(reducing by taking mean) words to one vector
				self.embedded_reviews = tf.reduce_mean(self.embedded_reviews, 1)
				print(self.embedded_reviews)


			# Create a dense connection layer for each layer size
			layers_sizes.insert(0, embedding_size)

			self.a_drop = self.embedded_reviews

			for i in range(len(layers_sizes) - 1):
				with tf.name_scope('dense-layer-%s-cells_num-%s' % (i, layers_sizes[i])):
					layer_shape = [layers_sizes[i], layers_sizes[i+1]]
					W = tf.Variable(tf.truncated_normal(layer_shape, stddev=0.1,), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[layer_shape[1]]), name="b")
					h = tf.nn.xw_plus_b(self.a_drop, W, b, name="h")
					a = tf.nn.relu(h, name='a_relu')
					with tf.name_scope('dropout-%s' % i):
						self.a_drop = tf.nn.dropout(a, self.dropout_keep_prob)


			# #include input size
			# for i, layer_size in enumerate(layers_sizes):
			# 	with tf.name_scope("dense-layer-%s-cells_num-%s" % (i,layer_size)):
			# 		layer_shape = [layer_size, embedding_size]
			# 		W = tf.Variable(tf.truncated_normal(layer_shape, stddev=0.1,), name="W"+str(i))
			# 		b = tf.Variable(tf.constant(0.1, shape=layer_shape[0]), name="b"+str(i))
			#
			# 		h = tf.nn.xw_plus_b(self.a_drop['a' + str(i - 1)], W, b, name="scores")
			# 		self.a_drop['a' + str(i)] = tf.nn.dropout(tf.nn.relu(h, name='relu'+str(i)), self.dropout_keep_prob, name="a"+str(i))
			# 		pooled_outputs.append(max_pooled)
			#
			#
			# # Add dropout
			# with tf.name_scope('dropout'):
			# 	self.a_drop = tf.nn.dropout(self.a_pool_flat, self.dropout_keep_prob)

			# Final (unnormalized) scores and predictions
			with tf.name_scope('output'):

				W = tf.Variable(tf.truncated_normal([layers_sizes[-1], num_classes], stddev=0.1,), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
				self.scores = tf.nn.xw_plus_b(self.a_drop, W, b, name="scores")
				self.predictions = tf.argmax(self.scores, 1, name='predictions')

			with tf.name_scope('loss'):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.Y)
				self.loss = tf.reduce_mean(losses)

			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")




