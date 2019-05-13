# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import data_helper as dp
import src.repository as repository
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_training_data():

    repo = repository.Repository()
    data = repo.get_data_set()
    number_of_training_data = len(data[np.where(data[:, 2] == 'training')])
    number_of_testing_data = len(data[np.where(data[:, 2] == 'testing')])

    undiacritized_char_set = data[:, 0]
    input_encoder = LabelEncoder()
    integer_encoded = input_encoder.fit_transform(undiacritized_char_set)

    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    input_onehot_encoded = one_hot_encoder.fit_transform(integer_encoded)

    diacritization_symbols = data[:, 1]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(diacritization_symbols)

    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    label_onehot_encoded = one_hot_encoder.fit_transform(integer_encoded)

    x_training = input_onehot_encoded[0: (number_of_training_data - 1):]
    x_testing = input_onehot_encoded[number_of_training_data::]

    y_training = label_onehot_encoded[0: (number_of_training_data - 1):]
    y_testing = label_onehot_encoded[number_of_training_data::]
    #y = dp.load_nn_labels_dataset_diacritics_only_string(training_dataset[:, [1]])

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
    dataX_train, dataY_train = get_training_data()
    dataX_test, dataY_test = get_testing_data()

    vocabulary, vocab_inverse, all_chars, dataset = create_vocab()
    n_chars, n_vocab = get_chars_and_vocab_count(vocabulary, all_chars)

    seq_length = 5
    X_train = []
    X_test = []

    for i in range(0, len(dataX_train) - seq_length, 1):
        seq_in = dataX_train[i:i + seq_length]
        X_train.append([vocabulary[char] for char in seq_in])

    for i in range(0, len(dataX_test) - seq_length, 1):
        seq_in = dataX_test[i:i + seq_length]
        X_test.append([vocabulary[char] for char in seq_in])

    X_train = numpy.array(X_train)
    X_test = numpy.array(X_test)
    Y_train = dataY_train[0: len(dataY_train) - 5]
    Y_test = dataY_test[0: len(dataY_test) - 5]

    n_patterns_train = len(X_train)
    n_patterns_test = len(X_test)

    # reshape X to be [samples, time steps, features]
    #X_train = numpy.reshape(numpy.array(X_train), (n_patterns_train, seq_length, 1))
    #X_test = numpy.reshape(numpy.array(X_test), (n_patterns_test, seq_length, 1))

    # normalize
    #X_train = X_train / float(n_vocab)
    #X_test = X_test / float(n_vocab)

    # input dim
    vocabulary_size = len(vocab_inverse)
    # output dim
    embedding_vector_length = 220

    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_vector_length, input_length=seq_length))

    model.add(Bidirectional(LSTM(350, return_sequences=True, name="BLSTM1")))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(350, name="BLSTM2")))
    model.add(Dropout(0.2))

    model.add(Dense(Y_train.shape[1], activation='softmax', name="dense"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # file_path = "weights.best.hdf5"
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    callbacks_list = [checkpoint, early_stop]

    # fit the model
    model.summary()
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              callbacks=callbacks_list, epochs=30, batch_size=128, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


