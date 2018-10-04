import logging
import myTools.moving_commands as mc
import os
from gensim.models import word2vec
import numpy as np
from tqdm import tnrange, tqdm_notebook

def create_word2Vec_model(sentences, num_features, min_word_count, num_workers, window_size, downsampling):
  """
  :param sentences:
  :param num_features:
  :param min_word_count:
  :param num_workers:
  :param window_size:
  :param downsampling:
  mc.move_to_model_location()
  model_dir = os.path.join(os.curdir, 'Word2Vec')
  os.chdir(model_dir)
  np.save(model_name + '_words_list', words_list)
  word_vectors = model.wv.syn0
  unk_vector = np.zeros([1, num_features])
  word_vectors = np.vstack([word_vectors, unk_vector])
  np.save(model_name + '_wordVectors', word_vectors)
  model.save(model_name)
  :return: model, model_name, words_list, word_vectors
  """
  logging.basicConfig(format='%(asctime)s : %%(levelname)s : %(message)s', level=logging.INFO)

  model = word2vec.Word2Vec(sentences,
                            workers=num_workers,
                            size=num_features,
                            min_count=min_word_count,
                            window=window_size,
                            sample=downsampling)
  model.init_sims(replace=True)
  model_name = str(num_features) + 'f_' + str(min_word_count) + 'minw_' + str(window_size) + 'window'
  words_list = model.wv.index2word

  mc.move_to_model_location()
  model_dir = os.path.join(os.curdir, 'Word2Vec')
  os.chdir(model_dir)
  np.save(model_name + '_words_list', words_list)
  word_vectors = model.wv.syn0
  unk_vector = np.zeros([1, num_features])
  word_vectors = np.vstack([word_vectors, unk_vector])
  np.save(model_name + '_wordVectors', word_vectors)
  model.save(model_name)

  return model, model_name, words_list, word_vectors

def load_word_list_and_words_vectors(model_name):
  """
	mc.move_to_model_location()
  model_dir = os.path.join(os.curdir, 'Word2Vec')
  os.chdir(model_dir)
	:param model_name:
	:return: word_list, word_vectors
	"""
  mc.move_to_model_location()
  model_dir = os.path.join(os.curdir, 'Word2Vec')
  os.chdir(model_dir)
  word_list = np.load(model_name + '_words_list.npy').tolist()
  word_vectors = np.load(model_name + '_wordVectors.npy')
  return word_list, word_vectors

def convert_pd_words_reviews_to_np_ids_matrix(reviews_df, maxSeqLength, wordsList, set_name):

	"""
	mc.move_to_data_location()
	np.save(set_name + '_ids_matrix', ids_matrix)
	np.save(set_name + '_sentiment_labels', sentiment_set)

	:param reviews_df:
	:param maxSeqLength:
	:param wordsList:
	:param set_name:
	:return: ids_matrix, sentiment_set
	"""
	number_of_rev = len(reviews_df)
	id_of_unknown = len(wordsList)
	ids_matrix = np.full((number_of_rev, maxSeqLength), id_of_unknown, dtype='int32')
	rev_set = reviews_df['review'].as_matrix()
	sentiment_set = reviews_df['sentiment'].as_matrix()
	rev_counter = 0
	word_counter = 0
	for i in tnrange(len(rev_set), desc='processing review'):
		for word in rev_set[i]:
			try:
				ids_matrix[rev_counter][word_counter] = wordsList.index(word)
			except ValueError:
				ids_matrix[rev_counter][word_counter] = id_of_unknown
			word_counter += 1
		rev_counter += 1
		word_counter = 0
	mc.move_to_data_location()
	np.save(set_name + '_ids_matrix', ids_matrix)
	np.save(set_name + '_sentiment_labels', sentiment_set)
	mc.move_to_main_location()
	return ids_matrix, sentiment_set

def load_ids_matrix_and_sentiment(x_filename, y_filename):
	mc.move_to_data_location()
	X = np.load(x_filename)
	Y = np.load(y_filename)
	Y = np.array(Y.tolist())
	return X, Y

#TODO : fun to load  the only right data ids_matrix

def translate_from_indx_matrix(intigerized_review, word_list):
	for i in intigerized_review:
		if i == len(word_list):
			print('UNK')
		else:
			print(word_list[i])



