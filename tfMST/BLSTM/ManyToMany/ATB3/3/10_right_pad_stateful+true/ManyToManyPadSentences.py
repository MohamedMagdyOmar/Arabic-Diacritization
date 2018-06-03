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
import DBHelperMethod
import os
import datetime
from keras import backend as K
# fix random seed for reproducibility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
time_steps = 1277
num_of_train_seqs = 10730
batch_size = 32


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
    training_sample_weight = []

    training_dataset = DBHelperMethod.load_dataset_by_type("training")
    #training_dataset = DBHelperMethod.load_dataset_by_type_and_sentence_number_for_testing_purpose("training", 1)
    sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("training")
    #sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("training")[0]

    labels_and_equiv_encoding = dp.get_label_table()
    input_one_hot_encoding = (dp.get_input_table())[:, 0]
    vocabulary, vocabulary_inv = dp.build_vocab(input_one_hot_encoding)

    start_time = datetime.datetime.now()
    for each_sentence_number in sentence_numbers:
        selected_sentence = training_dataset[numpy.where(training_dataset[:, 3] == str(each_sentence_number))]
        input = selected_sentence[:, 0]
        input_vocabed, output = prepare_input_and_output(input, vocabulary, selected_sentence[:, [0, 1]],
                                                         labels_and_equiv_encoding)

        training_sequence_list.append(input_vocabed)
        training_padded_output.append(output)

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)

    input_training_padded = pad_sequences(training_sequence_list, padding='pre')

    output_training_padded = pad_sequences(training_padded_output, padding='pre').astype(float)
    training_sample_weight = output_training_padded.sum(axis=2).astype(float)


    # testing data
    testing_sequence_list = []
    testing_padded_output = []
    testing_minor_sample_weight = []
    testing_sample_weight = []

    testing_dataset = DBHelperMethod.load_dataset_by_type("testing")
    #testing_dataset = DBHelperMethod.load_dataset_by_type_and_sentence_number_for_testing_purpose("testing", 1)
    sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("testing")
    #sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("testing")[0]

    start_time = datetime.datetime.now()
    for each_sentence_number in sentence_numbers:
        selected_sentence = testing_dataset[numpy.where(testing_dataset[:, 3] == str(each_sentence_number))]
        input = selected_sentence[:, 0]
        input_vocabed, output = prepare_input_and_output(input, vocabulary, selected_sentence[:, [0, 1]],
                                                         labels_and_equiv_encoding)

        testing_sequence_list.append(input_vocabed)
        testing_padded_output.append(output)

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)

    input_testing_padded = pad_sequences(testing_sequence_list, padding='pre', maxlen=input_training_padded.shape[1])

    output_testing_padded = pad_sequences(testing_padded_output, padding='pre', maxlen=output_training_padded.shape[1]).astype(float)
    testing_sample_weight = output_testing_padded.sum(axis=2).astype(float)

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
                        batch_input_shape=(batch_size, time_steps, 37),
                        input_length=X_train.shape[1], mask_zero=True))

    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True)))

    model.add(TimeDistributed(Dense(50, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'], sample_weight_mode='temporal')

    print(model.summary())

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    callbacks_list = [checkpoint, early_stop]

    model.fit(X_train, y_train, validation_data=(X_test, y_test, test_sample_weight),
              callbacks=callbacks_list, epochs=15, batch_size=32,
              verbose=1, sample_weight=train_sample_weight, shuffle=False)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    model.reset_states()
    print("Accuracy: %.2f%%" % (scores[1]*100))
