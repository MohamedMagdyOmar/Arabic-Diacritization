# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy
import data_helper as dp
import DBHelperMethod
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
# fix random seed for reproducibility
numpy.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_training_data():
    DBHelperMethod.connect_to_db()
    training_dataset = DBHelperMethod.load_dataset_by_type("training")

    x = dp.load_nn_input_dataset_string(training_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(training_dataset[:, [0, 1]])

    return x, y


def create_vocab():
    DBHelperMethod.connect_to_db()

    dataset = DBHelperMethod.load_data_set()
    chars = dp.load_nn_input_dataset_string(dataset[:, [0, 6]])
    vocab, vocab_inv = dp.build_vocab(chars)
    return vocab, vocab_inv, chars, dataset


def get_chars_and_vocab_count(vocab, chars):
    return len(chars), len(vocab)


if __name__ == "__main__":
    X_train, Y_train = get_training_data()
    vocabulary, vocab_inverse, all_chars, dataset = create_vocab()
    n_chars, n_vocab = get_chars_and_vocab_count(vocabulary, all_chars)

    seq_length = 3
    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = X_train[i:i + seq_length]
        dataX.append([vocabulary[char] for char in seq_in])

    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable

    print(X.shape[1])


