# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import data_helper as dp
import SukunCorrection
import FathaCorrection
import DictionaryCorrection
import DERCalculationHelperMethod
import WordLetterProcessingHelperMethod
import ExcelHelperMethod
import keras.models as models
from copy import deepcopy
import DBHelperMethod
import itertools
import os
# fix random seed for reproducibility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_row_1 = 0
current_row_2 = 0
Total_Error = 0
Total_Error_without_last_char = 0


def get_testing_data():
    DBHelperMethod.connect_to_db()
    testing_dataset = DBHelperMethod.load_dataset_by_type("testing")

    x = dp.load_nn_input_dataset_string_space_only(testing_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(testing_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(testing_dataset[:, [3]])

    testing_words = np.take(testing_dataset, 4, axis=1)
    input_testing_letters = np.take(testing_dataset, 0, axis=1)
    op_testing_letters = np.take(testing_dataset, 5, axis=1)
    sent_num = np.take(testing_dataset, 3, axis=1)
    letters_loc = np.take(testing_dataset, 6, axis=1)
    undiac_words = np.take(testing_dataset, 7, axis=1)

    return x, y, testing_words, input_testing_letters, op_testing_letters, sent_num, letters_loc, undiac_words


def get_all_undiac_words():
    return DBHelperMethod.get_all_un_diacritized_words_in_sentences()


def get_undiac_words_for_selected_sentence(list_of_all_words_and_sent_num, sentence_number):

    list_of_undiac_words_and_sent = list_of_all_words_and_sent_num[
        list(np.where(list_of_all_words_and_sent_num == str(sentence_number))[0])]

    return list_of_undiac_words_and_sent[:, 0].tolist()


def get_all_dic_words():
    return DBHelperMethod.get_dictionary()


def get_dic_words_for_selected_sentence(dic, undiac_words):

    dictionary = dic[:, 1]
    list_of_indices = []
    for each_word in undiac_words:
        list_of_indices.append([i for i, x in enumerate(dictionary) if x == each_word])

    indices = list(itertools.chain(*list_of_indices))
    return dic[indices]


def create_vocab():

    DBHelperMethod.connect_to_db()
    data_set = DBHelperMethod.load_data_set()
    chars = dp.load_nn_input_dataset_string_space_only(data_set[:, [0, 6]])
    vocab, vocab_inv = dp.build_vocab(chars)
    return vocab, vocab_inv, chars, data_set


def get_chars_and_vocab_count(vocab, chars):
    return len(chars), len(vocab)


if __name__ == "__main__":

    dataX_test, dataY_test, words, ip_letters, op_letters, sentences_num, loc, undiac_word = get_testing_data()

    vocabulary, vocab_inverse, all_chars, dataset = create_vocab()
    n_chars, n_vocab = get_chars_and_vocab_count(vocabulary, all_chars)
    dictionary = get_all_dic_words()

    seq_length = 5
    X_train = []
    X_test = []

    for i in range(0, len(dataX_test) - seq_length, 1):
        seq_in = dataX_test[i:i + seq_length]
        X_test.append([vocabulary[char] for char in seq_in])

    X_test = np.array(X_test)
    Y_test = dataY_test[0: len(dataY_test) - 5]
    ip_letters = ip_letters[0: len(ip_letters) - 5]
    op_letters = op_letters[0: len(op_letters) - 5]

    model = models.load_model('weights.017-0.8694.hdf5')

    print(model.summary())
    prediction = model.predict(X_test, verbose=1)

    nn_indices = prediction.argmax(axis=1)
    expected_indices = Y_test.argmax(axis=1)

    labels = dp.get_label_table()

    nn_labels = labels[nn_indices]
    nn_labels = np.take(nn_labels, 1, axis=1)
    expected_labels = labels[expected_indices]
    expected_labels = np.take(expected_labels, 1, axis=1)

    if len(nn_labels) == len(expected_labels) and len(nn_labels) == len(ip_letters):
        pass
    else:
        raise Exception("mismatch in number of elements in the array")

    # nn_op_letters = dp.concatenate_char_and_diacritization(ip_letters, nn_labels)
    expected_op_letters = op_letters

    list_of_sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by('testing')
    list_of_all_words_and_sent_num = get_all_undiac_words()

    current_sentence_counter = 0
    counter = 0
    start_range = 0
    end_range = 0
    for sentence_number in list_of_sentence_numbers:
        selected_sentence = DBHelperMethod.get_sentence_by(sentence_number)
        undiac_words = get_undiac_words_for_selected_sentence(list_of_all_words_and_sent_num, sentence_number)

        dic_words_for_selected_sent = get_dic_words_for_selected_sentence(dictionary, undiac_words)
        rnn_input = DBHelperMethod.get_un_diacritized_chars_by(sentence_number, 'testing')

        num_of_chars_in_selected_sent = len(rnn_input)

        end_range = num_of_chars_in_selected_sent + start_range

        nn_op_letters = dp.concatenate_char_and_diacritization(ip_letters[start_range: end_range: 1],
                                                               nn_labels[start_range: end_range: 1])

        expected_letters = expected_op_letters[start_range: end_range: 1]
        expected_letters = list(filter(lambda a: a != 'space', expected_letters))
        location = loc[start_range: end_range: 1]

        # Post Processing
        # nn_op_letters = list(filter(lambda a: a != 'spacespace', nn_op_letters))
        RNN_Predicted_Chars_And_Its_Location = dp.create_letter_location_object(nn_op_letters, location)
        RNN_Predicted_Chars_After_Sukun = SukunCorrection.sukun_correction(
            deepcopy(RNN_Predicted_Chars_And_Its_Location))
        RNN_Predicted_Chars_After_Fatha = FathaCorrection.fatha_correction(deepcopy(RNN_Predicted_Chars_After_Sukun))
        RNN_Predicted_Chars_After_Dictionary = DictionaryCorrection.get_diac_version_with_smallest_dist_no_db_access(
            RNN_Predicted_Chars_After_Fatha, undiac_words, dic_words_for_selected_sent)

        # Expected OP
        OP_Diac_Chars_Count = WordLetterProcessingHelperMethod.get_chars_count_for_each_word_in_this(
            selected_sentence)
        OP_Diac_Chars_And_Its_Location = WordLetterProcessingHelperMethod.get_location_of_each_char(
            expected_letters, OP_Diac_Chars_Count)
        OP_Diac_Chars_After_Sukun = SukunCorrection.sukun_correction(
            deepcopy(OP_Diac_Chars_And_Its_Location))

        # DER Calculation
        error = DERCalculationHelperMethod.get_diacritization_error \
            (RNN_Predicted_Chars_After_Dictionary, OP_Diac_Chars_After_Sukun, selected_sentence)

        error_without_last_letter = DERCalculationHelperMethod.get_diacritization_error_without_counting_last_letter \
            (RNN_Predicted_Chars_After_Dictionary, OP_Diac_Chars_After_Sukun, selected_sentence)

        # write error in excel file
        excel_1 = current_row_1
        current_row_1 = ExcelHelperMethod.write_data_into_excel_file(error, selected_sentence, excel_1)
        Total_Error += len(error)
        print("Total Error: ", Total_Error)

        excel_2 = current_row_2
        current_row_2 = ExcelHelperMethod.write_data_into_excel_file2(error_without_last_letter, selected_sentence,
                                                                      excel_2)
        Total_Error_without_last_char += len(error_without_last_letter)
        print("Total Error without Last Char: ", Total_Error_without_last_char)
        counter += 1
        print("we are now in sentence # ", counter)

        start_range = end_range


