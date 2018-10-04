import tensorflow as tf
import numpy as np

class TextProcessingCNN(object):
	def __init__(
		self,
	  max_sequence_length,
	  num_classes,
		word_vectors,
	  embedding_size,
	  filter_sizes,
	  num_filters,
		l2_reg_lambda=0.0):

			self.vocab_size = len(word_vectors)
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
			self.Y = tf.placeholder(tf.float32,
															[None, num_classes],
															name="Y")

			self.X = tf.placeholder(tf.int32,
			                       [None, max_sequence_length],
			                       name="X")
			with tf.name_scope("embedding"):
				#self.embedded_reviews = tf.Variable(tf.zeros([, max_sequence_length, embedding_size]), dtype=tf.float32, name="emmbeded_reviews")
				self.embedded_reviews = tf.nn.embedding_lookup(word_vectors, self.X)
				self.embedded_reviews = tf.cast(self.embedded_reviews, tf.float32)
				self.embedded_reviews_expanded = tf.expand_dims(self.embedded_reviews, -1)

			# Create a convolution + maxpool layer for each filter size
			pooled_outputs = []
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("convolution-maxpooling-%s" % filter_size):
					# Convolution layer
					filter_shape = [filter_size, embedding_size, 1, num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1,), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
					conv = tf.nn.conv2d(
															self.embedded_reviews_expanded,
															W,
															strides=[1,1,1,1],
															padding='VALID',
															name='convolution')
					# Apply nonlinearity
					a = tf.nn.relu(tf.nn.bias_add(conv, b, name='relu'))
					# maxpooling over the ouputs
					max_pooled = tf.nn.max_pool(
						a,
						ksize=[1, max_sequence_length - filter_size + 1, 1, 1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="max_pool"
					)
					pooled_outputs.append(max_pooled)

			# Combine all the pooled features
			num_filters_total = num_filters * len(filter_sizes)
			self.a_pool = tf.concat(pooled_outputs, 3)
			self.a_pool_flat = tf.reshape(self.a_pool, [-1, num_filters_total])

			# Add dropout
			with tf.name_scope('dropout'):
				self.a_drop = tf.nn.dropout(self.a_pool_flat, self.dropout_keep_prob)

			# Final (unnormalized) scores and predictions
			with tf.name_scope('output'):
				W = tf.get_variable(
					"W",
					shape=[num_filters_total, num_classes],
					initializer=tf.contrib.layers.xavier_initializer()
				)
				b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
				l2_loss = tf.nn.l2_loss(W)
				l2_loss += tf.nn.l2_loss(b)
				self.scores = tf.nn.xw_plus_b(self.a_drop, W, b, name="scores")
				self.predictions = tf.argmax(self.scores, 1, name='predictions')

			with tf.name_scope('loss'):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.Y)
				self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





