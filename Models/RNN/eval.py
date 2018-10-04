import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, 'C:\\Users\\Qbit\\Inzynierka')
import myTools.moving_commands as mc
from Models.Word2Vec import Word2vec as w2v
import Data.generate_batch as gb
import csv
import glob

# Data loading parameters
tf.flags.DEFINE_string("test_intigerized_reviews_file", "C:\\Users\\Qbit\\Inzynierka\\Data\\test_ids_matrix.npy", "Data source for test intigerized reviews")
tf.flags.DEFINE_string("test_sentiment_labels", "C:\\Users\\Qbit\\Inzynierka\\Data\\test_sentiment_labels.npy", "Data source for test sentiment labels")

# Eval Parameters
tf.flags.DEFINE_integer('batch_size', 200, 'Batch size (default 64)')

list_of_files = glob.glob('C:\\Users\\Qbit\\Inzynierka\\Models\\RNN\\runs\\*')
latest_file = max(list_of_files, key=os.path.getctime)
tf.flags.DEFINE_string('checkpoint_dir', latest_file + '\\checkpoints', 'Checkpoint directory from training run')
#tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device palacement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')

for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")



# if FLAGS.eval_train:
#
# 	# TODO : load test data
# 	pass
#
# else:
# 	pass
x_test, y_test = w2v.load_ids_matrix_and_sentiment(FLAGS.test_intigerized_reviews_file, FLAGS.test_sentiment_labels)
y_test = np.argmax(y_test, axis=1)
print('y_test: ', y_test)


print('\nEvaluationg...\n')

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement
	)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		# Get the placeholders from the graph by name
		# TODO: change name from batch_size_train_id_reviews to batch_size_id_reviews
		batch_size_train_matrix_id_reviews = graph.get_operation_by_name("X").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

		# Tensors we want to evaluate
		predictions = graph.get_operation_by_name("output/predictions").outputs[0]
		#Generate batches for one epoch
		batches = gb.batch_generator(list(x_test), FLAGS.batch_size, 1, shuffle=False)

		# Collect the predictions her
		all_predictions = []
		counter = 0
		for x_test_batch in batches:
			if len(x_test_batch) != FLAGS.batch_size:
				break
			print('batch_number: ',counter, end='\r', flush=True)
			batch_predictions = sess.run(predictions, {batch_size_train_matrix_id_reviews: x_test_batch, dropout_keep_prob: 1.0})
			all_predictions = np.concatenate([all_predictions, batch_predictions])
			counter += 1

		# Print accuracy if y_test is defined
		y_test = y_test[:len(all_predictions)]
		correct_predictions = float(sum(all_predictions == y_test))
		print("Total number of test examples: {}".format(len(y_test)))
		print('Accuracy: {:g}'.format(correct_predictions/float(len(y_test))))

		# Save the evaluation to a csv
		mc.move_to_data_location()
		# TODO: change when learn new model and prepere new data
		#test_set = pd.read_csv('test_set.csv')
		test_set = pd.read_csv('test_processed_set.csv')
		x_raw = test_set['review'][:len(all_predictions)]
		predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
		# without dots
		out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
		print("Saving evaluation to  {0}".format(out_path))
		with open(out_path, 'w') as f:
			csv.writer(f).writerows(predictions_human_readable)

		acc_path = os.path.join(FLAGS.checkpoint_dir, "accuracy.csv")
		print("Saving accurancy to  {0}".format(acc_path))
		with open(acc_path, 'w') as f:
			f.write('{}'.format(correct_predictions/float(len(y_test))))


