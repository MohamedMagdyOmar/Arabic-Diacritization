# this same as "test", but here you make a window of 5 chars, and if actual sentence is finished so you pad
# the sequence as per the github post explained, so this is considered as modification for "test" file

import numpy
import data_helper as dp
import SequenceProcessing as sqp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import DBHelperMethod
import os
# fix random seed for reproducibility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

window_size = 5
sequence_length = window_size


def load_training_data():
    dp.establish_db_connection()
    training_dataset = DBHelperMethod.load_dataset_by_type("testing")

    x = dp.load_nn_input_dataset_string_space_only(training_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(training_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(training_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.extract_sent_and_pad(x, sen_len, window_size)
    padded_output = dp.extract_sent_and_pad_output(y, sen_len, window_size)

    return sentences_padded, padded_output, vocabulary, vocabulary_inv


def load_testing_data():
    dp.establish_db_connection()
    training_dataset = DBHelperMethod.load_dataset_by_type("testing")

    x = dp.load_nn_input_dataset_string_space_only(training_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(training_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(training_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.extract_sent_and_pad(x, sen_len, window_size)
    padded_output = dp.extract_sent_and_pad_output(y, sen_len, window_size)

    return sentences_padded, padded_output, vocabulary, vocabulary_inv


def check_key_exist(vocab_training, vocab_testing):

    if any(key in vocab_training for key in vocab_testing):
        pass
    else:
        raise Exception("keys are missing")


if __name__ == "__main__":

    X_train, y_train, vocabulary_train, vocabulary_inv_train = load_training_data()
    X_test, y_test, vocabulary_test, vocabulary_inv_test = load_testing_data()
    check_key_exist(vocabulary_train, vocabulary_test)

    y_train = y_train.reshape(len(y_train), 5, 51)

    y_test = y_test.reshape(len(y_test), 5, 51)
    # input dim
    vocabulary_size = len(vocabulary_inv_train)
    # output dim
    embedding_vector_length = 51

    model = Sequential()

    model.add(Embedding(vocabulary_size, embedding_vector_length, input_length=sequence_length))

    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

    model.add(TimeDistributed(Dense(51, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    callbacks_list = [checkpoint, early_stop]

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              callbacks=callbacks_list, epochs=40, batch_size=64, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
