import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
sys.path.insert(0, 'C:\\Users\\Qbit\\Inzynierka')
from myTools import  preprocessingData as  pred
from Models.Word2Vec import Word2vec as w2v
import myTools.moving_commands as mc
import Data.generate_batch as gb
from Models.CNN_Text import TextProcessingCNN as Cnn

# Data loading params
# TODO : loading parameters of directory where the data are
tf.flags.DEFINE_float('dev_sample_percentage', .1, 'Percetage of the training data to use for validation')
tf.flags.DEFINE_string("train_dev_intigerized_reviews_file", "C:\\Users\\Qbit\\Inzynierka\\Data\\train_dev_ids_matrix.npy", "Data source for train/dev intigerized reviews")
tf.flags.DEFINE_string("train_dev_sentiment_labels", "C:\\Users\\Qbit\\Inzynierka\\Data\\train_dev_sentiment_labels.npy", "Data source for train/dev sentiment labels")
tf.flags.DEFINE_string("word2Vec_model", "300f_40minw_10window", "Source for pretrained word2vec model")

# Model Nyperparameters

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_hights", "3,4,5", "Comma-seprated filter sizes (default: '3,4,5'), how many word are convoluted with a filter")
tf.flags.DEFINE_integer("filter_width", 300, "Number of width, usually equal to embedding_dim, wa went to filer whole word")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters

tf.flags.DEFINE_integer("train_batch_size", 32, "Train batch size (default: 32)")
tf.flags.DEFINE_integer("dev_batch_size", 500, "Train batch size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs(default: 2)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")


# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))

# Data Preparation
# ==============================================


# Load data
x, y = w2v.load_ids_matrix_and_sentiment(FLAGS.train_dev_intigerized_reviews_file, FLAGS.train_dev_sentiment_labels)
# TODO: loading data, load id_matrix , load preprocessed train_dev_split and then make cross validation

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print('x_train : ', x_train.shape)
print('y_train : ', y_train.shape)
word_list, word_vectors = w2v.load_word_list_and_words_vectors(FLAGS.word2Vec_model)
print('Vocabulary Size: {:d}'.format(len(word_list)))
print('Train/Dev split: {:d}/{:d}'.format(len(y_train), len(y_dev)))
dev_batch_size = len(y_dev)

# Training
# ==============================================

if __name__ == '__main__':
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement=FLAGS.allow_soft_placement,
			log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = Cnn.TextProcessingCNN(max_sequence_length = x_train.shape[1],
								num_classes=len(y_train[0]),
								word_vectors=word_vectors,
								embedding_size=FLAGS.embedding_dim,
								filter_sizes=list(map(int, FLAGS.filter_hights.split(','))),
								num_filters=FLAGS.filter_width,
								l2_reg_lambda=FLAGS.l2_reg_lambda)

		#Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Keep track of gradient values and sparsity (optional)
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)

		# Output directory for models and summaries
		# TODO: create such directory that it will be good meh
		timestamp = str(int(time.time()))
		mc.move_to_main_location()
		model_name = "cnn_fh" + str(FLAGS.filter_hights) + "_fw" + str(FLAGS.filter_width) + '_drop' + str(FLAGS.dropout_keep_prob) + '_bs' + str(FLAGS.train_batch_size)
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "Models", "CNN_Text", "runs", timestamp + model_name))
		print('Writing to {}\n'.format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.summary.scalar("loss", cnn.loss)
		acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

		# Train summaries
		train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		# Checkpoint direcotory, Tensorflow assumes this direcotyr already exitst so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		# TODO: check if this is right directory
		checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		def train_step(x_batch, y_batch):
			feed_dict = {
				cnn.X: x_batch,
				cnn.Y: y_batch,
				cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
			}
			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy), end='\r', flush=True)
			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_batch, y_batch, writer=None):
			"""
			Evaluates model on a dev set
			"""
			feed_dict = {
				cnn.X: x_batch,
				cnn.Y: y_batch,
				cnn.dropout_keep_prob: 1.0,
			}
			step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("Evaluation: {}: step {}, loss{:g}, acc {:g}".format(time_str, step, loss, accuracy), end='\r', flush=True)
			if writer:
				writer.add_summary(summaries, step)

		# Generate batches
		batches = gb.batch_generator(list(zip(x_train, y_train)), FLAGS.train_batch_size, FLAGS.num_epochs)
		dev_batches = gb.batch_generator(list(zip(x_dev, y_dev)), FLAGS.dev_batch_size, FLAGS.num_epochs * 10 )
		# Training loop. For each batch..

		for batch in batches:
			x_batch, y_batch = zip(*batch)
			train_step(x_batch, y_batch)
			current_step = tf.train.global_step(sess, global_step) - 1
			if current_step % FLAGS.evaluate_every == 0:
				x_dev_batch, y_dev_batch = zip(*dev_batches.__next__())
				dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
				print("")

			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path), end='\r', flush=True)

		path = saver.save(sess, checkpoint_prefix, global_step=tf.train.global_step(sess, global_step))
		x_dev_batch, y_dev_batch = zip(*dev_batches.__next__())
		dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
		print("Saved model checkpoint to {}\n".format(path), end='\r', flush=True)







