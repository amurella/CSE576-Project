from pickle import load
import numpy as np
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
	model.add(LSTM(n_units))
	#model.add(Dropout(0.4))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	#model.add(Dropout(0.4))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# create empty arrays to contain batch of features and labels
def generator(data, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, batch_size):
	batch_features = np.zeros((batch_size, spa_length))
	batch_labels = np.zeros((batch_size, eng_length, eng_vocab_size))
	while True:
		#get indices for random subset of data
		#print("data.shape[0]: ")
		#print(data.shape[0])
		indices = np.random.choice(data.shape[0], batch_size, replace=False)
		#print("indices.shape: ")
		#print(indices.shape)
		#print("data[indices].shape: ")
		#print(data[indices].shape)
		#batch_features = data[indices][0]
		batch_features = data[indices, 0]
		batch_labels = data[indices, 1]
		#print("batch_features shape: ")
		#print(batch_features.shape)
		#print("batch_labels shape: ")
		#print(batch_labels.shape)

		#process data
		trainX = encode_sequences(spa_tokenizer, spa_length, batch_features)
		trainY = encode_sequences(eng_tokenizer, eng_length, batch_labels)
		trainY = encode_output(trainY, eng_vocab_size)
		#batch_features[i] = some_processing(features[index])
		#batch_labels[i] = labels[index]
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
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
train_batch_size = 8000
test_batch_size = 880
model.fit_generator(generator(train, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, train_batch_size), samples_per_epoch=1, nb_epoch=20, validation_data=generator(test, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, test_batch_size), validation_steps=1, callbacks=[checkpoint], verbose=2)
