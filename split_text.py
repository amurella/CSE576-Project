from math import floor
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('dataset/english-spanish.pkl')

# get dataset size
n_sentences = len(raw_dataset[:,1])
n_sentences = 30000
n_train = floor(n_sentences*0.9)
n_train = 27000
n_test = n_sentences-n_train
dataset = raw_dataset[:n_sentences, :]

# random shuffle
shuffle(dataset)

# split into train/test
# 90% train, 10% test
train, test = dataset[:n_train], dataset[n_train:]

# save
save_clean_data(dataset, 'dataset/english-spanish-both.pkl')
save_clean_data(train, 'dataset/english-spanish-train.pkl')
save_clean_data(test, 'dataset/english-spanish-test.pkl')
