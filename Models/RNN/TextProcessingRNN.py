import tensorflow as tf

class TextProcessingRNN(object):
	def __init__(
		self,
		max_sequence_length,
		batch_size,
		lstm_units,
		num_classes,
		word_vectors,
		embedding_size
		):
			self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
			self.Y = tf.placeholder(
															tf.float32,
			                        [batch_size, num_classes],
			                        name='Y')

			# X is a batch size id matrix
			self.X = tf.placeholder(
															tf.int32,
			                        [batch_size, max_sequence_length],
			                        name='X')

			with tf.name_scope('embedding'):
				self.embedded_reviews = tf.Variable(tf.zeros([batch_size, max_sequence_length, embedding_size]))
				self.embedded_reviews = tf.nn.embedding_lookup(word_vectors, self.X)
				self.embedded_reviews = tf.cast(self.embedded_reviews, tf.float32)

			with tf.name_scope('LSTM_cells_layer'):
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
				lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_keep_prob)
					#lstm_cell = tf.nn.dropout(lstm_cell, self.dropout_keep_prob)
				self.value, _ = tf.nn.dynamic_rnn(lstm_cell, self.embedded_reviews, dtype=tf.float32)

			with tf.name_scope('output'):
				W = tf.Variable(
					tf.truncated_normal([lstm_units, num_classes]),
					name='W'
					)
				b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b'))
				self.value = tf.transpose(self.value, [1, 0, 2])
				self.last_value = tf.gather(self.value, int(self.value.get_shape()[0]) - 1)
				self.scores = tf.nn.xw_plus_b(self.last_value, W, b, name='scores')
				self.predictions = tf.argmax(self.scores, 1, name='predictions')

			with tf.name_scope('accuracy'):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

			with tf.name_scope('loss'):
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.Y))







