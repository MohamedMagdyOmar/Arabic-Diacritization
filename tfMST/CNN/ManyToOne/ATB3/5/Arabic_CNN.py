# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy
import data_helper as dp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import TensorBoard
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import DBHelperMethod
import os
# fix random seed for reproducibility
numpy.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
req_char_index = 13
window_size = 17


def load_training_data():
    dp.establish_db_connection()
    training_dataset = DBHelperMethod.load_dataset_by_type("training")

    # x = dp.load_nn_input_dataset_string(training_dataset[:, [0, 6]])
    x = dp.load_nn_input_dataset_string_space_only(training_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(training_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(training_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.pad_sentences1(x, sen_len, req_char_index, window_size)

    return sentences_padded, y, vocabulary, vocabulary_inv


def load_testing_data():
    dp.establish_db_connection()
    testing_dataset = DBHelperMethod.load_dataset_by_type("testing")

    # x = dp.load_nn_input_dataset_string(testing_dataset[:, [0, 6]])
    x = dp.load_nn_input_dataset_string_space_only(testing_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(testing_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(testing_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.pad_sentences1(x, sen_len, req_char_index, window_size)

    return sentences_padded, y, vocabulary, vocabulary_inv


def check_key_exist(vocab_training, vocab_testing):

    if any(key in vocab_training for key in vocab_testing):
        pass
    else:
        raise Exception("keys are missing")


if __name__ == "__main__":

    X_train, y_train, vocabulary_train, vocabulary_inv_train = load_training_data()
    X_train = (numpy.arange(X_train.max()) == X_train[..., None] - 1).astype(int)

    X_test, y_test, vocabulary_test, vocabulary_inv_test = load_testing_data()
    X_test = (numpy.arange(X_test.max()) == X_test[..., None] - 1).astype(int)

    check_key_exist(vocabulary_train, vocabulary_test)

    model = Sequential()

    model.add(Conv1D(filters=256, kernel_size=17, padding='same', activation='relu', input_shape=(window_size, 37)))
    model.add(Conv1D(filters=128, kernel_size=17, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=17, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(250, dropout=0.2, return_sequences=True, activation='sigmoid')))
    model.add(Flatten())
    model.add(Dropout(0.2))

    #model.add(Flatten())
    model.add(Dense(50, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    callbacks_list = [checkpoint, early_stop]

    tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              callbacks=callbacks_list, epochs=40, batch_size=79, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))