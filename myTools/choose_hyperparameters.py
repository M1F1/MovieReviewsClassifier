import os
import sys
from numpy import random as rnd
from numpy import genfromtxt
import time
import pandas as pd
import sys
sys.path.insert(0, 'C:\\Users\\Qbit\\Inzynierka')
from myTools import  moving_commands as mc
import glob
from random import randrange
import random
import subprocess
import random

mc.move_to_model_location()

# system arguments
nn = sys.argv[1]
epochs_num = int(sys.argv[2])
sample_num = int(sys.argv[3])
second_run = bool(int(sys.argv[4]))
print('second_run', second_run)


if not second_run:
	batch_size = [20 * rnd.random_integers(1, 20) for _ in range(sample_num) ]
	print('batch_size', batch_size)
	dropout_keep_prob = [round(rnd.rand(), 3) for _ in range(sample_num)]
	print('dropout_keep_prob', dropout_keep_prob)
	learning_rate = [pow(10, -6 * rnd.rand()) for _ in range(sample_num)]
	print('learning_rate ', learning_rate)

if nn == 'MLP':
	# second run
	if second_run:
		print('Second run')
		batch_size = [20 * rnd.random_integers(10, 17) for _ in range(sample_num) ]
		print('batch_size', batch_size)
		dropout_keep_prob = [round(random.uniform(0.7, 1.), 3) for _ in range(sample_num)]
		print('dropout_keep_prob', dropout_keep_prob)
		learning_rate = [pow(10, -6 * random.uniform(0.4, 0.6)) for _ in range(sample_num)]
		print('learning_rate ', learning_rate)

	print('IN MLP')
	cwd = os.getcwd()
	model_dir = os.path.abspath(os.path.join(cwd, 'MLP'))
	os.chdir(model_dir)
	hyper_search_dir = os.path.join(model_dir, 'hyperparameters_search')
	print('hyper_seach_dir:', hyper_search_dir)
	runs_dir = os.path.join(model_dir, 'runs')

	# special hyperparameters for MLP
	if not second_run:
		max_number_of_layers = 5
		min_number_of_layers = 1
		max_size_of_layer = 500
		min_size_of_layer = 1
	else:
		max_number_of_layers = 5
		min_number_of_layers = 3
		max_size_of_layer = 450
		min_size_of_layer = 150

	layers_num = [rnd.random_integers(min_number_of_layers, max_number_of_layers) for _ in range(sample_num)]

	time_stamp = str(int(time.time()))
	values = []
	print('i am in:', os.getcwd())
	print(os.listdir())
	for i in range(sample_num):
		print('')
		print('Example: ', i)
		print('')

		num_layer = layers_num[i]
		layers_sizes = [rnd.random_integers(min_size_of_layer, max_size_of_layer) for _ in range(num_layer)]
		l_sizes = ','.join(map(str, layers_sizes))

		start_time = time.time()
		subprocess.call(['python', 'train.py', '--train_batch_size=' + str(batch_size[i]),
		                 '--num_epochs=' + str(epochs_num),
		                 '--dropout_keep_prob=' + str(dropout_keep_prob[i]),
		                 '--learning_rate=' + str(round(learning_rate[i], 4)),
		                 '--layers_sizes=' + l_sizes], shell=True)
		elapsed_training_time = time.time() - start_time

		start_time = time.time()
		subprocess.call(['python', 'eval.py', '--batch_size=' + str(batch_size[i])], shell=True)
		elapsed_testing_time = time.time() - start_time

		model_name = 'bs_' + str(batch_size[i]) +\
								 '_dkp_' + str(dropout_keep_prob[i]) +\
								 '_lr_' + str(learning_rate[i]) +\
								 '_ls_' + str(l_sizes)
		list_of_files = glob.glob(runs_dir + '\\*')
		latest_file = max(list_of_files, key=os.path.getctime)
		accuracy = genfromtxt(os.path.join(runs_dir, latest_file, 'checkpoints', 'accuracy.csv'))
		values.append([accuracy, model_name, elapsed_training_time, elapsed_testing_time])

	df = pd.DataFrame(values, columns=['accuracy', 'model_name', 'elapsed_training_time', 'elapsed_testing_time'])
	df.to_csv(os.path.join(hyper_search_dir, time_stamp + '.csv'), index=False)

	print(df)

elif nn == 'CNN':
	if second_run:
		print('Second run')
		batch_size = [20 * rnd.random_integers(14, 17) for _ in range(sample_num) ]
		print('batch_size', batch_size)
		dropout_keep_prob = [round(random.uniform(0.4, 0.6), 3) for _ in range(sample_num)]
		print('dropout_keep_prob', dropout_keep_prob)
		learning_rate = [pow(10, -6 * random.uniform(0.4, 0.6)) for _ in range(sample_num)]
		print('learning_rate ', learning_rate)

	print('IN CNN')
	cwd = os.getcwd()
	model_dir = os.path.abspath(os.path.join(cwd, 'CNN_Text'))
	os.chdir(model_dir)
	hyper_search_dir = os.path.join(model_dir, 'hyperparameters_search')
	if not os.path.exists(hyper_search_dir):
				os.makedirs(hyper_search_dir)
	print('hyper_seach_dir:', hyper_search_dir)
	runs_dir = os.path.join(model_dir, 'runs')

	# special hyperparameters for CNN
	max_conv_layer_num = 3
	min_conv_layer_num = 1
	min_filter_size = 1
	max_filter_size = 5
	min_filter_width = 1
	max_filter_width = 300
	if second_run:
		max_conv_layer_num = 2
		min_conv_layer_num = 1
		min_filter_size = 3
		max_filter_size = 5
		min_filter_width = 200
		max_filter_width = 300

	conv_layers_num = [rnd.random_integers(min_conv_layer_num, max_conv_layer_num) for _ in range(sample_num)]
	print('conv_layers_num: ', conv_layers_num)
	filter_width = [rnd.random_integers(min_filter_width, max_filter_width) for _ in range(sample_num)]
	print('filter_widths: ', filter_width)

	time_stamp = str(int(time.time()))
	values = []
	print('i am in:', os.getcwd())
	print(os.listdir())
	for i in range(sample_num):
		num_layer = conv_layers_num[i]
		filter_sizes = [rnd.random_integers(min_filter_size, max_filter_size) for _ in range(num_layer)]
		f_sizes = ','.join(map(str, filter_sizes))
		print('')
		print('Example: ', i)
		print('')
		start_time = time.time()
		subprocess.call(['python', 'train.py', '--train_batch_size=' + str(batch_size[i]),
		                 '--num_epochs=' + str(epochs_num),
		                 '--dropout_keep_prob=' + str(dropout_keep_prob[i]),
		                 '--learning_rate=' + str(round(learning_rate[i], 4)),
		                 '--filter_hights=' + f_sizes,
		                 '--filter_width=' + str(filter_width[i])],
		                  shell=True)
		elapsed_training_time = time.time() - start_time

		start_time = time.time()
		subprocess.call(['python', 'eval.py', '--batch_size=' + str(batch_size[i])], shell=True)
		elapsed_testing_time = time.time() - start_time

		model_name = 'bs_' + str(batch_size[i]) +\
								 '_dkp_' + str(dropout_keep_prob[i]) +\
								 '_lr_' + str(learning_rate[i]) +\
								 '_fs_' + str(f_sizes) +\
								 '_fw_' + str(filter_width[i])
		list_of_files = glob.glob(runs_dir + '\\*')
		latest_file = max(list_of_files, key=os.path.getctime)
		accuracy = genfromtxt(os.path.join(runs_dir, latest_file, 'checkpoints', 'accuracy.csv'))
		values.append([accuracy, model_name, elapsed_training_time, elapsed_testing_time])

	df = pd.DataFrame(values, columns=['accuracy', 'model_name', 'elapsed_training_time', 'elapsed_testing_time'])
	df.to_csv(os.path.join(hyper_search_dir, time_stamp + '.csv'), index=False)

	print(df)

elif nn == 'RNN':
	if second_run:
		print('Second run')
		batch_size = [20 * rnd.random_integers(1, 1) for _ in range(sample_num) ]
		print('batch_size', batch_size)
		dropout_keep_prob = [round(random.uniform(0.8, 0.9), 3) for _ in range(sample_num)]
		print('dropout_keep_prob', dropout_keep_prob)
		#learning_rate = [pow(10, -6 * random.uniform(0.5, 0.6)) for _ in range(sample_num)]
		learning_rate = [20e-5]
		print('learning_rate ', learning_rate)
	print('IN RNN')
	cwd = os.getcwd()
	model_dir = os.path.abspath(os.path.join(cwd, 'RNN'))
	os.chdir(model_dir)
	hyper_search_dir = os.path.join(model_dir, 'hyperparameters_search')
	if not os.path.exists(hyper_search_dir):
				os.makedirs(hyper_search_dir)
	print('hyper_search_dir:', hyper_search_dir)
	runs_dir = os.path.join(model_dir, 'runs')

	# special hyperparameters for RNN

	lstm_units = [pow(2, random.randint(0, 8)) for _ in range(sample_num)]
	if second_run:
		lstm_units = [pow(2, random.randint(8, 8)) for _ in range(sample_num)]
	print('LSTM_UNITS=', lstm_units)
	time_stamp = str(int(time.time()))
	values = []
	print('i am in:', os.getcwd())
	print(os.listdir())
	for i in range(sample_num):
		print('')
		print('Example: ', i)
		print('')

		start_time = time.time()
		subprocess.call(['python', 'train.py', '--train_batch_size=' + str(batch_size[i]),
		                 '--num_epochs=' + str(epochs_num),
		                 '--dropout_keep_prob=' + str(dropout_keep_prob[i]),
		                 '--learning_rate=' + str(round(learning_rate[i], 4)),
		                 '--lstm_units=' + str(lstm_units[i])],
		                  shell=True)
		elapsed_training_time = time.time() - start_time

		start_time = time.time()
		subprocess.call(['python', 'eval.py', '--batch_size=' + str(batch_size[i])], shell=True)
		elapsed_testing_time = time.time() - start_time

		model_name = 'bs_' + str(batch_size[i]) +\
								 '_dkp_' + str(dropout_keep_prob[i]) +\
								 '_lr_' + str(learning_rate[i]) +\
								 '_lstm_u_' + str(lstm_units[i])
		list_of_files = glob.glob(runs_dir + '\\*')
		latest_file = max(list_of_files, key=os.path.getctime)
		accuracy = genfromtxt(os.path.join(runs_dir, latest_file, 'checkpoints', 'accuracy.csv'))
		values.append([accuracy, model_name, elapsed_training_time, elapsed_testing_time])

	df = pd.DataFrame(values, columns=['accuracy', 'model_name', 'elapsed_training_time', 'elapsed_testing_time'])
	df.to_csv(os.path.join(hyper_search_dir, time_stamp + '.csv'), index=False)

	print(df)


