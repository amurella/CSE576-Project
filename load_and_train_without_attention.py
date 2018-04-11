from pickle import load
import numpy as np
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units, activation='softsign', dropout=0.0, recurrent_dropout=0.0))
	model.add(RepeatVector(tar_timesteps))
	#model.add(AttentionDecoder(n_units, tar_vocab))
	model.add(LSTM(n_units, activation='softsign', dropout=0.0, recurrent_dropout=0.0, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# create empty arrays to contain batch of features and labels
def generator(data, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, batch_size):
	batch_features = np.zeros((batch_size, spa_length))
	batch_labels = np.zeros((batch_size, eng_length, eng_vocab_size))
	while True:

		#shuffle data each time we run through it
		print("Shuffling Data")
		np.random.shuffle(data)
		"""
		print("data.shape: ")
		print(data.shape)
		print("data: ")
		print(data)
		"""

		num_batches = math.ceil(data.shape[0] / batch_size)
		for i in range(0, num_batches):

			#print progress every 500 batches
			if i % 500 == 0:
				print("Generating batch %d" % i)

			start = i*batch_size
			end = (i+1)*batch_size
			batch_features = data[start:end, 1]
			batch_labels = data[start:end, 0]
			"""
			print("batch_features.shape: ")
			print(batch_features.shape)
			print("batch_features: ")
			print(batch_features)
			print("batch_labels.shape: ")
			print(batch_labels.shape)
			print("batch_labels: ")
			print(batch_labels)
			"""

			#reshape arrays
			#batch_features = np.reshape(batch_features, (batch_size, spa_length))
			#batch_labels = np.reshape(batch_labels, (batch_size, eng_length, eng_vocab_size))
			
			#process data
			trainX = encode_sequences(spa_tokenizer, spa_length, batch_features)
			trainY = encode_sequences(eng_tokenizer, eng_length, batch_labels)
			#batch_features[i] = some_processing(features[index])
			#batch_labels[i] = labels[index]
			"""
			print("trainX.shape: ")
			print(trainX.shape)
			print("trainX: ")
			print(trainX)
			print("trainY.shape: ")
			print(trainY.shape)
			print("trainY: ")
			print(trainY)
			"""			

			#convert to onehot (for cross-entropy loss function)
			trainY = encode_output(trainY, eng_vocab_size)

			yield trainX, trainY


# load datasets
dataset = load_clean_sentences('dataset/english-spanish-both.pkl')
train = load_clean_sentences('dataset/english-spanish-train.pkl')
test = load_clean_sentences('dataset/english-spanish-test.pkl')
print("dataset shape: ")
print(dataset.shape)
print("train shape: ")
print(train.shape)
print("train shape DIM=0: ")
print(train.shape[0])
print("test shape: ")
print(test.shape)

print("TRAIN data: ")
print(train)

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
#eng_tokenizer = create_tokenizer(generator)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare spanish tokenizer
spa_tokenizer = create_tokenizer(dataset[:, 1])
spa_vocab_size = len(spa_tokenizer.word_index) + 1
spa_length = max_length(dataset[:, 1])
print('Spanish Vocabulary Size: %d' % spa_vocab_size)
print('Spanish Max Length: %d' % (spa_length))

# prepare training data
#trainX = encode_sequences(spa_tokenizer, spa_length, train[:, 1])
#trainX, trainY = encode_sequences(spa_tokenizer, spa_length, spa_generator)
#print("trainX shape: ")
#print(trainX.shape)
#print("train[:, 1] shape: ")
#print(train[:, 1].shape)
#trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
#trainY = encode_output(trainY, eng_vocab_size)
#print("trainY shape: ")
#print(trainY.shape)

# prepare validation data
#testX = encode_sequences(spa_tokenizer, spa_length, test[:, 1])
#testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
#testY = encode_output(testY, eng_vocab_size)

# define model
model = define_model(spa_vocab_size, eng_vocab_size, spa_length, eng_length, 256)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
num_training_samples = train.shape[0]
num_testing_samples = test.shape[0]
train_batch_size = 64
test_batch_size = 64
model.fit_generator(generator(train, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, train_batch_size), steps_per_epoch=num_training_samples/train_batch_size, epochs=30, validation_data=generator(test, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, test_batch_size), validation_steps=num_testing_samples/test_batch_size, callbacks=[checkpoint], verbose=2)


