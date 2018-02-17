# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import data_helper as dp
from keras.models import load_model
import os
# fix random seed for reproducibility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_testing_data():
    dp.establish_db_connection()
    testing_dataset = dp.load_dataset_by_type("testing")

    x = dp.load_nn_input_dataset_string(testing_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(testing_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(testing_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.pad_sentences(x, sen_len, 4, 10)

    testing_words = np.take(testing_dataset, 4, axis=1)
    input_testing_letters = np.take(testing_dataset, 0, axis=1)
    op_testing_letters = np.take(testing_dataset, 5, axis=1)
    sent_num = np.take(testing_dataset, 3, axis=1)
    letters_loc = np.take(testing_dataset, 6, axis=1)
    undiac_word = np.take(testing_dataset, 7, axis=1)

    return sentences_padded, y, vocabulary, vocabulary_inv, testing_words, input_testing_letters, op_testing_letters,\
           sent_num, letters_loc, undiac_word


def create_nn_op_words(letters, location):
    x = 1


if __name__ == "__main__":

    X_test, y_test, vocabulary_test, vocabulary_inv_test, words, ip_letters, op_letters, sentences_num, loc, \
    undiac_word = load_testing_data()

    model = load_model('weights.003-0.6761.hdf5')
    print(model.summary())
    prediction = model.predict(X_test, verbose=1)

    nn_indices = prediction.argmax(axis=1)
    expected_indices = y_test.argmax(axis=1)

    labels = dp.get_label_table()

    nn_labels = labels[nn_indices]
    nn_labels = np.take(nn_labels, 1, axis=1)
    expected_labels = labels[expected_indices]
    expected_labels = np.take(expected_labels, 1, axis=1)

    if len(nn_labels) == len(expected_labels) and len(nn_labels) == len(ip_letters):
        pass
    else:
        raise Exception("mismatch in number of elements in the array")

    nn_op_letters = np.core.defchararray.add(ip_letters, nn_labels)
    expected_op_letters = op_letters
    create_nn_op_words(nn_op_letters, loc)
    g = 1



    print(prediction)

