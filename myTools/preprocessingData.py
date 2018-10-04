from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
from random import shuffle
import re
import nltk
from nltk.corpus import stopwords
import csv
import myTools.moving_commands as mc
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tnrange, tqdm_notebook

def review2pddataframe(path_to_csv, sentiment):
	lastdir = os.getcwd()
	print('I am in : %s' % (lastdir))
	os.chdir(path_to_csv)
	curd = os.getcwd()
	print('Changing to:  %s' % (curd))
	filenames = os.listdir(curd)
	size = len(filenames)
	counter = 0
	rows = []
	for filename in filenames:
		id, rest = filename.split('_')
		rating = rest.split('.')[0]
		with open(filename, 'r', encoding='utf8') as file:
			soup = BeautifulSoup(file.read())
			review = soup.get_text()
		row = (review, int(rating), int(sentiment))
		rows.append(row)
		if(counter + 1) % 5000 == 0:
			print('adding %d review out of %d : \n %s' % (counter + 1, size, review))
		counter = counter + 1

	print('ending...')
	os.chdir(lastdir)
	print('Changing back to : %s' % (lastdir))

	df = pd.DataFrame(data = rows, columns=['review', 'rating', 'sentiment'])
	return df

def gather_sentiment_data_set():
	"""
	labelled_data.to_csv('all_labelled_data_before_processing.csv', encoding="utf-8", index=False)
	unlabelled_data.to_csv('all_unlabelled_data_before_processing.csv', encoding="utf-8", index=False)

	:return: labelled_data, unlabelled_data
	"""
	mc.move_to_main_location()
	path_to_train_pos = os.path.abspath(os.path.join(os.path.curdir, "SentimentDataSet", "train", "pos"))
	path_to_train_neg = os.path.abspath(os.path.join(os.path.curdir, "SentimentDataSet", "train", "neg"))
	path_to_test_pos = os.path.abspath(os.path.join(os.path.curdir, "SentimentDataSet", "test", "pos"))
	path_to_test_neg = os.path.abspath(os.path.join(os.path.curdir, "SentimentDataSet", "test", "neg"))

	train_pos_df = review2pddataframe(path_to_train_pos, 1)
	train_neg_df = review2pddataframe(path_to_train_neg, 0)
	test_pos_df = review2pddataframe(path_to_test_pos, 1)
	test_neg_df = review2pddataframe(path_to_test_neg, 0)

	labelled_data = [test_neg_df, train_neg_df, test_pos_df, train_pos_df]
	labelled_data = pd.concat(labelled_data)
	labelled_data = labelled_data.sample(frac=1).reset_index(drop=True)

	path_to_unlabelled_data = os.path.abspath(os.path.join(os.path.curdir, "SentimentDataSet", "train", "unsup"))
	unlabelled_data = review2pddataframe(path_to_unlabelled_data, -1)
	unlabelled_data = unlabelled_data.sample(frac=1).reset_index(drop=True)

	mc.move_to_data_location()
	labelled_data.to_csv('all_labelled_data_before_processing.csv', encoding="utf-8", index=False)
	unlabelled_data.to_csv('all_unlabelled_data_before_processing.csv', encoding="utf-8", index=False)

	return labelled_data, unlabelled_data

def preprocessing_reviews_in_df_to_sentences(data_before_processing, filename):
	"""

	write_list_of_lists_to_csv(sentences, filename)
	:param data_before_processing:
	:param filename:
	:return: sentences (python list)
	"""
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	reviews = data_before_processing['review'].as_matrix().tolist()
	reviews_sentences =[tokenizer.tokenize(review.strip()) for review in reviews]
	clean = lambda x: re.sub('[^a-zA-Z1-9\s]', '', x)
	lower_split = lambda x: x.lower().split()
	clean_sentences = [list(map(clean, reviews_sentences[i])) for i in range(len(reviews_sentences))]
	to_low_split_sentences = [list(map(lower_split, clean_sentences[i])) for i in range(len(clean_sentences))]
	sentences = [sentence for review in to_low_split_sentences for sentence in review]

	mc.move_to_data_location()
	write_list_of_lists_to_csv(sentences, filename)
	return sentences

def preprocessing_reviews_in_df_to_words(data_before_processing, filename):
	"""
	mc.move_to_data_location()
	write_list_of_lists_to_csv(processed_reviews, filename)
	:param data_before_processing:
	:param filename:
	:return: processed_reviews
	"""
	reviews = data_before_processing['review'].as_matrix().tolist()
	clean = lambda x: re.sub('[^a-zA-Z1-9\s]', '', x)
	processed_reviews = [(''.join(list(map(clean, reviews[i]))).lower()).split() for i in range(len(reviews))]
	mc.move_to_data_location()
	write_list_of_lists_to_csv(processed_reviews, filename)
	return processed_reviews

def create_shuffle_save_data(array_of_dfs, csvfilename):
	print('.... Concatenating dfs .... ')
	data = pd.concat(array_of_dfs)
	print('.... Shuffling rows ....')
	data = data.sample(frac=1).reset_index(drop=True)
	print('.... Saving df to : %s ....' % (csvfilename))
	data.to_csv(csvfilename, encoding='utf-8', index=False)
	return data



def write_list_of_lists_to_csv(ll, filename):
	print('.... Writing data to %s ....' % filename)
	with open(filename, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(ll)
	print('.... DONE ....')

def read_csv_to_list_of_lists(filename):
	print('.... Reading data from %s ....' % filename)
	with open(filename, 'rU') as f:
		reader = csv.reader(f)
		data = list(list(rec) for rec in csv.reader(f, delimiter=','))
	print('.... DONE ....')
	return data



def preprocessing_reviews_in_df(set_df, max_seq_length, filename):
	"""
	mc.move_to_data_location()
	set_df.to_csv(filename + '.csv', index=False)
	:param set_df:
	:param max_seq_length:
	:param filename:
	:return: set_df
	"""

	# BUG when reading from csv -- quotes
	for i in tnrange(1, desc='processing'):
		set_df = set_df[['review', 'sentiment']]
		set_df.info()
		set_df['review'] = set_df['review'].apply(lambda x: re.sub('[^a-zA-Z1-9\s]', '', x))
		set_df['review'] = set_df['review'].apply(lambda x: x.lower().split()[:max_seq_length])
		set_df['sentiment'] = set_df['sentiment'].apply(lambda x: [1, 0] if x == 1 else [0, 1])
		mc.move_to_data_location()
		set_df.to_(filename + '.csv', index=False)
	return set_df


def pick_sample(all_data, sample_size):
    X = list(all_data.index.values)
    Y = all_data['rating']
    sss = StratifiedShuffleSplit(n_splits=1, test_size = sample_size, random_state=0)
    gen_object = sss.split(X, Y)
    indices = [*gen_object][0]
    rest_indx, sample_indx = indices[0], indices[1]
    rest_set = all_data.loc[all_data.index.isin(rest_indx)]
    sample_set = all_data.loc[all_data.index.isin(sample_indx)]
    return [sample_set,rest_set]

def split_into_train_dev_test_sets(all_data, train_size):
    train_set, dev_test_set = pick_sample(train_size, all_data)
    test_set, dev_set = pick_sample(0.5, dev_test_set)
    return [train_set, dev_set, test_set]


