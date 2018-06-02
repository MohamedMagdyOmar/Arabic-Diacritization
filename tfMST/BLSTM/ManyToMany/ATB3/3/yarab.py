# this same as "test", but here you make a window of 5 chars, and if actual sentence is finished so you pad
# the sequence as per the github post explained, so this is considered as modification for "test" file

import numpy
import data_helper as dp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros, newaxis
from itertools import chain
import DBHelperMethod
import os
import datetime
from keras import backend as K
# fix random seed for reproducibility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 32
num_of_steps = 40


def prepare_input_and_output(input, vocab, output, label_encoding):
    each_sequence_labels = []
    input_after_vocabed = []
    for each_input, each_output_row in zip(input, output):

        # for input
        input_after_vocabed.append(vocab[each_input])

        # for output
        if each_output_row[1] != '':
            raw_input_data = each_output_row[1]
        else:
            raw_input_data = each_output_row[0]

        index_of_raw_label_data = numpy.where(label_encoding == raw_input_data)

        if numpy.size(index_of_raw_label_data) != 0:
            label = label_encoding[index_of_raw_label_data[0], 2][0]
            label = list(map(int, label))
            each_sequence_labels.append(label)
        else:
            Exception('label not found')

    return input_after_vocabed, each_sequence_labels


def load_data():
    max_length = 1280
    dp.establish_db_connection()
    training_sequence_list = []
    training_padded_output = []

    training_dataset = DBHelperMethod.load_dataset_by_type("training")
    sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("training")

    labels_and_equiv_encoding = dp.get_label_table()
    input_one_hot_encoding = (dp.get_input_table())[:, 0]
    vocabulary, vocabulary_inv = dp.build_vocab(input_one_hot_encoding)

    start_time = datetime.datetime.now()
    for each_sentence_number in sentence_numbers:
        selected_sentence = training_dataset[numpy.where(training_dataset[:, 3] == str(each_sentence_number))]
        input = selected_sentence[:, 0]
        input_vocabed, output = prepare_input_and_output(input, vocabulary, selected_sentence[:, [0, 1]],
                                                         labels_and_equiv_encoding)

        input_vocabed = numpy.array(input_vocabed)
        number_of_zeros = max_length - len(input_vocabed)
        input_vocabed = numpy.pad(input_vocabed, (number_of_zeros, 0), 'constant', constant_values=0)
        input_vocabed = numpy.reshape(input_vocabed, (-1, num_of_steps))

        training_sequence_list.append(input_vocabed)

        output = numpy.array(output)
        output = numpy.pad(output, ((number_of_zeros, 0), (0, 0)), 'constant', constant_values=0)
        output = output.reshape(batch_size, num_of_steps, 51)

        training_padded_output.append(output)

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)
    training_sequence_list = list(chain(*training_sequence_list))
    training_sequence_list = numpy.array(training_sequence_list)

    training_padded_output = list(chain(*training_padded_output))
    training_padded_output = numpy.array(training_padded_output)

    input_training_padded = training_sequence_list
    output_training_padded = training_padded_output
    training_sample_weight = 1

    # testing data
    testing_sequence_list = []
    testing_padded_output = []

    testing_dataset = DBHelperMethod.load_dataset_by_type("testing")
    sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("testing")

    start_time = datetime.datetime.now()
    for each_sentence_number in sentence_numbers:
        selected_sentence = testing_dataset[numpy.where(testing_dataset[:, 3] == str(each_sentence_number))]
        input = selected_sentence[:, 0]
        input_vocabed, output = prepare_input_and_output(input, vocabulary, selected_sentence[:, [0, 1]],
                                                         labels_and_equiv_encoding)

        input_vocabed = numpy.array(input_vocabed)
        number_of_zeros = max_length - len(input_vocabed)
        input_vocabed = numpy.pad(input_vocabed, (number_of_zeros, 0), 'constant', constant_values=0)
        input_vocabed = numpy.reshape(input_vocabed, (-1, num_of_steps))

        testing_sequence_list.append(input_vocabed)

        output = numpy.array(output)
        output = numpy.pad(output, ((number_of_zeros, 0), (0, 0)), 'constant', constant_values=0)
        output = output.reshape(batch_size, num_of_steps, 51)
        testing_padded_output.append(output)

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)
    testing_sequence_list = list(chain(*testing_sequence_list))
    testing_sequence_list = numpy.array(testing_sequence_list)

    testing_padded_output = list(chain(*testing_padded_output))
    testing_padded_output = numpy.array(testing_padded_output)

    input_testing_padded = testing_sequence_list
    output_testing_padded = testing_padded_output

    testing_sample_weight = 1

    return input_training_padded, output_training_padded, training_sample_weight, vocabulary, vocabulary_inv\
        , input_testing_padded, output_testing_padded, testing_sample_weight


if __name__ == "__main__":

    X_train, y_train, train_sample_weight, vocabulary_train, vocabulary_inv_train, X_test, y_test, test_sample_weight \
        = load_data()

    # input dim
    vocabulary_size = len(vocabulary_inv_train) + 1  # (+1 because we have 0 for masking)

    # output dim
    embedding_vector_length = 37

    model = Sequential()

    model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_vector_length,
                        input_length=X_train.shape[1], mask_zero=True))

    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

    model.add(TimeDistributed(Dense(51, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'], sample_weight_mode='temporal')

    print(model.summary())

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    callbacks_list = [checkpoint, early_stop]

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              callbacks=callbacks_list, epochs=15, batch_size=32,
              verbose=1)
    '''
    for each_epoch in range(0, 10):
        for input, output in zip(X_train, y_train):
            input = input[newaxis, :]
            output = output[newaxis, :, :]
            model.train_on_batch(input, output)

        for input, output in zip(X_test, y_test):
            input = input[newaxis, :]
            output = output[newaxis, :, :]
            model.test_on_batch(input, output)

    # Final evaluation of the model
    '''
    scores = model.evaluate(X_test, y_test, verbose=0)
    model.reset_states()
    print("Accuracy: %.2f%%" % (scores[1]*100))
