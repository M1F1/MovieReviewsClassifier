import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from tensorflow.contrib.tensorboard.plugins import projector
import os
import subprocess
import sys
#model_name = sys.argv[1]
cur_dir = os.getcwd()
model = Word2Vec.load(os.path.join(cur_dir, '300f_40minw_10window'))
words_list = model.wv.index2word
word_vectors = model.wv.syn0
dim = len(word_vectors[1])
vocab_size = len(words_list)
print('dim: ', dim)
print('vocab: ', vocab_size)

w2v = np.zeros((vocab_size, dim))
print(w2v.shape)

projector_dir = os.path.join(cur_dir, 'projector')
if not os.path.exists(projector_dir):
	os.makedirs(projector_dir)
meta_data_path = os.path.join(projector_dir, 'prefix_metadata.tsv')
with open(meta_data_path, 'w+') as file_metadata:
	for i, word in enumerate(model.wv.index2word):
		w2v[i] = model[word]
		file_metadata.write(word + '\n')


sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
	embedding = tf.Variable(w2v, trainable=False, name='embeddings')

tf.global_variables_initializer().run()
saver = tf.train.Saver()
writer = tf.summary.FileWriter(projector_dir, sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embeddings'
embed.metadata_path = meta_data_path

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)
saver.save(sess, os.path.join(projector_dir, 'w2v_model.ckpt'), global_step=10000)
print(projector_dir)
subprocess.call(['tensorboard', '--logdir=' + str(projector_dir)], shell=True)



