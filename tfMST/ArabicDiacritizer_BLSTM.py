# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy
import data_helper as dp
import DBHelperMethod
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
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


def get_testing_data():
    DBHelperMethod.connect_to_db()
    training_dataset = DBHelperMethod.load_dataset_by_type("testing")

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
    dataX_train, dataY_train = get_training_data()
    dataX_test, dataY_test = get_testing_data()

    vocabulary, vocab_inverse, all_chars, dataset = create_vocab()
    n_chars, n_vocab = get_chars_and_vocab_count(vocabulary, all_chars)

    seq_length = 3
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
    Y_train = dataY_train[0: len(dataY_train) - 3]
    Y_test = dataY_test[0: len(dataY_test) - 3]

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

    model.add(Bidirectional(LSTM(256, return_sequences=True, name="BLSTM1")))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(256, name="BLSTM2")))
    model.add(Dropout(0.2))

    model.add(Dense(Y_test.shape[1], activation='softmax', name="dense"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # file_path = "weights.best.hdf5"
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    callbacks_list = [checkpoint, early_stop]

    # fit the model
    model.summary()
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              callbacks=callbacks_list, epochs=50, batch_size=64, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


