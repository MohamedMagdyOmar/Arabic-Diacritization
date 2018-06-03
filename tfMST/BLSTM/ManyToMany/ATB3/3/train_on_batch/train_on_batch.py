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
from keras.models import save_model, load_model
import DBHelperMethod
import os
import datetime
import tempfile
from keras import backend as K
# fix random seed for reproducibility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

        training_sequence_list.append(numpy.array(input_vocabed))
        training_padded_output.append(numpy.array(output))

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)
    '''
    input_training_padded = pad_sequences(training_sequence_list, padding='pre')
    output_training_padded = pad_sequences(training_padded_output, padding='pre').astype(float)
    training_sample_weight = output_training_padded.sum(axis=2).astype(float)
    '''
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

        testing_sequence_list.append(numpy.array(input_vocabed))
        testing_padded_output.append(numpy.array(output))

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)

    '''
    input_testing_padded = pad_sequences(testing_sequence_list, padding='pre', maxlen=input_training_padded.shape[1])
    output_testing_padded = pad_sequences(testing_padded_output, padding='pre', maxlen=output_training_padded.shape[1]).astype(float)    
    testing_sample_weight = output_testing_padded.sum(axis=2).astype(float)
    '''

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
                        input_length=None))

    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

    model.add(TimeDistributed(Dense(50, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    patience = 5
    best_loss = 1e6
    rounds_without_improvement = 0
    training_batch_counter = 0
    testing_batch_counter = 0

    for each_epoch in range(0, 10):
        losses_list = list()
        training_batch_counter = 0
        testing_batch_counter = 0

        for x, y in zip(X_train, y_train):
            training_batch_counter += 1
            x = x[numpy.newaxis, :]
            y = y[numpy.newaxis, :, :]

            loss = model.train_on_batch(x, y)
            print("epoch number: ", each_epoch, ", training_batch_num: ", training_batch_counter, " loss: ", loss)

        for x, y in zip(X_test, y_test):
            testing_batch_counter += 1
            x = x[numpy.newaxis, :]
            y = y[numpy.newaxis, :, :]
            model.test_on_batch(x, y)
            loss = model.test_on_batch(x, y)
            losses_list.append(loss[0])
            print("epoch number: ", each_epoch, ", testing_batch_num: ", testing_batch_counter, " loss: ", loss)
        c = sum(losses_list)
        mean_loss = sum(losses_list) / len(losses_list)

        if mean_loss < best_loss:
            best_loss = mean_loss
            rounds_without_improvement = 0
            _, fname = tempfile.mkstemp('.h5')
            save_model(model, fname)

            print("loss improved")
        else:
            rounds_without_improvement += 1
            print("No Improvement")

        if rounds_without_improvement == patience:
            print("patience finished !!")
            break

    model.reset_states()

