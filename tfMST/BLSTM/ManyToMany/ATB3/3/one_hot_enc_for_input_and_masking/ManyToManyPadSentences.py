import numpy
import data_helper as dp
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import DBHelperMethod
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def prepare_input_and_output(input, input_encoding, output, label_encoding):
    each_sequence_labels = []
    each_sequence_inputs = []
    for each_input, each_output_row in zip(input, output):

        # for input
        raw_input_data = each_input
        index_of_raw_input_data = numpy.where(input_encoding == raw_input_data)
        if numpy.size(index_of_raw_input_data) != 0:
            input = input_encoding[index_of_raw_input_data[0], 1][0]
            input = list(map(int, input))
            each_sequence_inputs.append(input)
        else:
            Exception('input not found')

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

    return each_sequence_inputs, each_sequence_labels


def load_data():
    dp.establish_db_connection()
    training_sequence_list = []
    training_padded_output = []

    training_dataset = DBHelperMethod.load_dataset_by_type("training")
    sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("training")

    labels_and_equiv_encoding = dp.get_label_table()
    input_one_hot_encoding = dp.get_input_table()

    start_time = datetime.datetime.now()
    for each_sentence_number in sentence_numbers:
        selected_sentence = training_dataset[numpy.where(training_dataset[:, 3] == str(each_sentence_number))]
        input = selected_sentence[:, 0]
        input_vocabed, output = prepare_input_and_output(input, input_one_hot_encoding, selected_sentence[:, [0, 1]],
                                                         labels_and_equiv_encoding)

        training_sequence_list.append(input_vocabed)
        training_padded_output.append(output)

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)

    input_training_padded = pad_sequences(training_sequence_list, padding='post').astype(float)

    output_training_padded = pad_sequences(training_padded_output, padding='post').astype(float)
    training_sample_weight = output_training_padded.sum(axis=2).astype(float)

    # testing data
    testing_sequence_list = []
    testing_padded_output = []

    testing_dataset = DBHelperMethod.load_dataset_by_type("testing")
    sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("testing")

    start_time = datetime.datetime.now()
    for each_sentence_number in sentence_numbers:
        selected_sentence = testing_dataset[numpy.where(testing_dataset[:, 3] == str(each_sentence_number))]
        input = selected_sentence[:, 0]
        input_vocabed, output = prepare_input_and_output(input, input_one_hot_encoding, selected_sentence[:, [0, 1]],
                                                         labels_and_equiv_encoding)

        testing_sequence_list.append(input_vocabed)
        testing_padded_output.append(output)

    end_time = datetime.datetime.now()
    print("prepare data takes : ", end_time - start_time)

    input_testing_padded = pad_sequences(testing_sequence_list, padding='post', maxlen=input_training_padded.shape[1])\
        .astype(float)

    output_testing_padded = pad_sequences(testing_padded_output, padding='post', maxlen=output_training_padded.shape[1])\
        .astype(float)

    testing_sample_weight = output_testing_padded.sum(axis=2).astype(float)

    return input_training_padded, output_training_padded, training_sample_weight\
        , input_testing_padded, output_testing_padded, testing_sample_weight


if __name__ == "__main__":

    X_train, y_train, train_sample_weight, X_test, y_test, test_sample_weight = load_data()

    model = Sequential()

    model.add(Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])))
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

    model.fit(X_train, y_train, validation_data=(X_test, y_test, test_sample_weight),
              callbacks=callbacks_list, epochs=15, batch_size=32,
              verbose=1, sample_weight=train_sample_weight)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    model.reset_states()
    print("Accuracy: %.2f%%" % (scores[1]*100))
