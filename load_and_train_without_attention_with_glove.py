
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
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L1L2
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines, max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

def create_embedding_matrix(tokenizer, embedding_filename, vocab_size, embedding_dim=100):

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(embedding_filename)

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    # exit()
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# encode and pad sequences
def encode_sequences(tokenizer, length, lines, vocab_size):

    # print(lines)
    # print(np.shape(lines))
    # exit()

    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')

    return X  # one hot encode target sequence


def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# define NMT model
def define_model(source_vocab_size, target_vocab_size, src_timesteps, tar_timesteps, src_embedding_matrix, src_embedding_dim=50, n_units=200):
    model = Sequential()
    print(src_timesteps)
    print(tar_timesteps)
    #model.add(Embedding(source_vocab_size, src_embedding_dim, input_length=src_timesteps, mask_zero=True))
    model.add(Embedding(source_vocab_size, src_embedding_dim, weights=[src_embedding_matrix], input_length=src_timesteps, trainable=False))
    
    #why??
    #n_units = tar_embedding_dim
    model.add(LSTM(n_units, activation='softsign', dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=L1L2(l1=0.00, l2=0.0001)))
    model.add(RepeatVector(tar_timesteps))
    # model.add(AttentionDecoder(n_units, tar_vocab))
    model.add(LSTM(n_units, activation='softsign', dropout=0.5, recurrent_dropout=0.5, return_sequences=True, kernel_regularizer=L1L2(l1=0.00, l2=0.0001)))
    model.add(TimeDistributed(Dense(target_vocab_size, activation='softmax')))
    return model


# create empty arrays to contain batch of features and labels
def generator(data, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, batch_size):
    batch_features = np.zeros((batch_size, spa_length))
    batch_labels = np.zeros((batch_size, eng_length, eng_vocab_size))
    while True:

        # shuffle data each time we run through it
        print("Shuffling Data")
        np.random.shuffle(data)
        """
        print("data.shape: ")
        print(data.shape)
        print("data: ")
        print(data)
        """

        num_batches = math.ceil(data.shape[0] / batch_size)
        print("num_batches: ")
        print(num_batches)
        for i in range(0, num_batches):

            # print progress every 100 batches
            if i % 100 == 0:
                print("Generating batch %d" % i)

            start = i * batch_size
            end = (i + 1) * batch_size
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

            # reshape arrays
            # batch_features = np.reshape(batch_features, (batch_size, spa_length))
            # batch_labels = np.reshape(batch_labels, (batch_size, eng_length, eng_vocab_size))

            # process data
            trainX = encode_sequences(spa_tokenizer, spa_length,  batch_features, eng_vocab_size)
            trainY = encode_sequences(eng_tokenizer, eng_length, batch_labels, eng_vocab_size)
            # batch_features[i] = some_processing(features[index])
            # batch_labels[i] = labels[index]
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

            # convert to onehot (for cross-entropy loss function)
            trainY = encode_output(trainY, eng_vocab_size)

            yield trainX, trainY


#limit GPU memory so we can train multiple models simultaneously
def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

# load datasets
dataset = load_clean_sentences('dataset/english-spanish-both.pkl')
# dataset = load_sentences_for_embedding('dataset/english-spanish-both.pkl')

"""
print('VIA OLD METHOD')
print('\nPrinting full dataset')
print(dataset)
print('\nPrinting English sentences only')
print(list(dataset[:, 0]))
"""

train = load_clean_sentences('dataset/english-spanish-train.pkl')
test = load_clean_sentences('dataset/english-spanish-test.pkl')
#print('\nPrinting Training Data')
#print(train)

print("dataset shape: ")
print(dataset.shape)
print("train shape: ")
print(train.shape)
print("train shape DIM=0: ")
print(train.shape[0])
print("test shape: ")
print(test.shape)

#print("TRAIN data: ")
#print(train)

# prepare english tokenizer
eng_max_words = 25000
eng_tokenizer = create_tokenizer(dataset[:, 0], eng_max_words)
# eng_tokenizer = create_tokenizer(generator)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

#SELECT THE EMBEDDING FILE TO USE:

#50dim embedding, 6bil words, wikipedia dataset
# ENG_EMBEDDING_FILE = './glove_wikipedia/glove.6B.50d.txt'
# ENG_EMBEDDING_DIM = 50

#300dim embedding, 840bil words
# ENG_EMBEDDING_FILE = './glove_840/glove.840B.300d.txt'
# ENG_EMBEDDING_DIM = 300

#300dim embedding, 840bil words
#ENG_EMBEDDING_FILE = './embeddings/glove.6B/glove.6B.300d.txt'
#ENG_EMBEDDING_DIM = 300

#eng_embedding_matrix = create_embedding_matrix(eng_tokenizer, ENG_EMBEDDING_FILE, eng_vocab_size, embedding_dim=ENG_EMBEDDING_DIM)

#print('PRINTING ENG EMBEDDING MATRIX')
#print(eng_embedding_matrix)
# exit()

print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

# prepare spanish tokenizer
spa_max_words = 40000
spa_tokenizer = create_tokenizer(dataset[:, 1], spa_max_words)
spa_vocab_size = len(spa_tokenizer.word_index) + 1
spa_length = max_length(dataset[:, 1])
print('Spanish Vocabulary Size: %d' % spa_vocab_size)
print('Spanish Max Length: %d' % (spa_length))

SPA_EMBEDDING_FILE = './embeddings/word2vec.1b/SBW-vectors-300-min5.txt'
SPA_EMBEDDING_DIM = 300

spa_embedding_matrix = create_embedding_matrix(spa_tokenizer, SPA_EMBEDDING_FILE, spa_vocab_size, embedding_dim=SPA_EMBEDDING_DIM)

#print('PRINTING SPA EMBEDDING MATRIX')
#print(spa_embedding_matrix)

# prepare training data
# trainX = encode_sequences(spa_tokenizer, spa_length, train[:, 1])
# trainX, trainY = encode_sequences(spa_tokenizer, spa_length, spa_generator)
# print("trainX shape: ")
# print(trainX.shape)
# print("train[:, 1] shape: ")
# print(train[:, 1].shape)
# trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
# trainY = encode_output(trainY, eng_vocab_size)
# print("trainY shape: ")
# print(trainY.shape)

# prepare validation data
# testX = encode_sequences(spa_tokenizer, spa_length, test[:, 1])
# testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
# testY = encode_output(testY, eng_vocab_size)

# define model
model = define_model(spa_vocab_size, eng_vocab_size, spa_length, eng_length, spa_embedding_matrix, src_embedding_dim=SPA_EMBEDDING_DIM, n_units=200)
# rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# summarize defined model
print(model.summary())
plot_model(model, to_file='model_primary.png', show_shapes=True)
# fit model
filename = 'model_primary.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
num_training_samples = train.shape[0]
print("num_training_samples: ")
print(num_training_samples)
num_testing_samples = test.shape[0]
train_batch_size = 16
test_batch_size = 16
train_batches_per_epoch = num_training_samples / train_batch_size
print("train_batches_per_epoch")
print(train_batches_per_epoch)
test_batches_per_epoch = num_testing_samples / test_batch_size
print("test_batches_per_epoch")
print(test_batches_per_epoch)

model.fit_generator(
	generator(train, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, train_batch_size),
	steps_per_epoch=num_training_samples / train_batch_size,
	epochs=30,
	validation_data=generator(test, spa_tokenizer, eng_tokenizer,
	spa_length, eng_length, eng_vocab_size, test_batch_size),
	validation_steps=num_testing_samples / test_batch_size,
	max_queue_size=2,
	callbacks=[checkpoint], verbose=2)
# def generator(data, spa_tokenizer, eng_tokenizer, spa_length, eng_length, eng_vocab_size, batch_size):
