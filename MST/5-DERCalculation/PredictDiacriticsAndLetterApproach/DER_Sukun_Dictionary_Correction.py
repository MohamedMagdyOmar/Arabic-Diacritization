# -*- coding: utf-8 -*-

import csv
import xlsxwriter
import MySQLdb
import MySQLdb.cursors
import os
import glob
import unicodedata
from xlrd import open_workbook
from xlutils.copy import copy
from copy import deepcopy
import locale


class LetterPosition:
    letter = "",
    location = ""

    def __init__(self):
        self.letter = ""
        self.location = ""


class ErrorDetails:
    actual_letter = "",
    expected_letter = "",
    error_location = 0,
    word = ""

    def __init__(self):
        self.actual_letter = ""
        self.expected_letter = ""
        self.error_location = 0
        self.word = ""

total_error = 0
total_error_without_last_letter = 0
total_chars_including_un_diacritized_target_letter = 0
total_chars_not_including_un_diacritized_target_letter = 0

current_row_excel_1 = 0
current_row_excel_2 = 0

current_row_in_excel_file = 1
extension = 'csv'

path = 'D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample'
diacritization_error_excel_file_path = "D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample\Errors" \
                                       "\Book1.xls "

diacritization_error_without_last_letter_excel_file_path = "D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample\Errors" \
                                       "\Book2.xls "

workbook = xlsxwriter.Workbook(diacritization_error_excel_file_path)
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'Actual')
worksheet.write(0, 1, 'Expected')
worksheet.write(0, 2, 'Error Location')
worksheet.write(0, 3, 'word contain error')
workbook.close()

workbook2 = xlsxwriter.Workbook(diacritization_error_without_last_letter_excel_file_path)
worksheet2 = workbook2.add_worksheet()
worksheet2.write(0, 0, 'Actual')
worksheet2.write(0, 1, 'Expected')
worksheet2.write(0, 2, 'Error Location')
worksheet2.write(0, 3, 'word contain error')
workbook2.close()


def connect_to_db():
    db = MySQLdb.connect(host="127.0.0.1",  # your host, usually localhost
                         user="root",  # your username
                         passwd="Islammega88",  # your password
                         db="mstdb",  # name of the data base
                         cursorclass=MySQLdb.cursors.SSCursor,
                         use_unicode=True,
                         charset="utf8",
                         init_command='SET NAMES UTF8')
    global cur
    cur = db.cursor()


def get_list_of_sentence_numbers():
    sentence_number_of_testing_query = "select distinct SentenceNumber from parseddocument where LetterType='testing' "\
                                       "order by idCharacterNumber asc "

    cur.execute(sentence_number_of_testing_query)

    sentence_numbers = (cur.fetchall())
    sentence_numbers = [each_number[0] for each_number in sentence_numbers]
    sentence_numbers = list(map(int, sentence_numbers))
    return sentence_numbers


def get_sentence_from_db(counter, __list_of_sentence_numbers):

    current_sentence_number = __list_of_sentence_numbers[counter]
    connect_to_db()
    selected_sentence_query = "select Word from listofwordsandsentencesineachdoc where SentenceNumber = " + \
                              str(current_sentence_number)

    cur.execute(selected_sentence_query)

    current_sentence = cur.fetchall()
    # current_sentence = sorted(set(current_sentence), key=lambda x: current_sentence.index(x))
    current_sentence = [eachTuple[0] for eachTuple in current_sentence]
    # current_sentence = [x[0] for x in groupby(current_sentence)]

    return current_sentence, current_sentence_number


def read_csv_file_of_a_predicted_sentence(filename):
    path_of_file = 'D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample\\' + filename
    rnn_output_of_current_sentence = []
    with open(path_of_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:
            rnn_output_of_current_sentence.append(map(float, row))

        return rnn_output_of_current_sentence


def get_neurons_numbers_with_highest_output_value(rnn_output):
    list_0f_neurons_indices_with_highest_value = []
    # Take care !!!, that the array is zero index
    for row in rnn_output:
        list_0f_neurons_indices_with_highest_value.append(row.index(max(row)))

    return list_0f_neurons_indices_with_highest_value


def get_all_letters_from_db():
    list_of_all_diacritized_letters_query = "select DiacritizedCharacter from diaconehotencoding"

    cur.execute(list_of_all_diacritized_letters_query)

    db_diacritized_letters = cur.fetchall()
    db_diacritized_letters = [eachTuple[0] for eachTuple in db_diacritized_letters]

    return db_diacritized_letters


def get_actual_letters(list_of_all_diacritized_letters, neurons_locations_with_highest_output):

    list_of_actual_letters_before_sukun_correction = []
    for neuron_location in neurons_locations_with_highest_output:
        # Take care, we make +1 because "neurons_locations_with_highest_output" is zero index
        list_of_actual_letters_before_sukun_correction.append(list_of_all_diacritized_letters[neuron_location - 1])

    return list_of_actual_letters_before_sukun_correction


def get_expected_letters(__sentence_number):

    list_of_expected_diacritized_letters_query = "select DiacritizedCharacter from parseddocument where " \
                                                 "LetterType='testing' and SentenceNumber = " + str(__sentence_number)

    cur.execute(list_of_expected_diacritized_letters_query)

    db_expected_letters_before_sukun_correction = cur.fetchall()
    db_expected_letters_before_sukun_correction = \
        [eachTuple[0] for eachTuple in db_expected_letters_before_sukun_correction]

    return db_expected_letters_before_sukun_correction


def sukun_correction(list_of_actual_letters_before_sukun_correction, list_of_expected_letters_before_sukun_correction):

    list_of_actual_letters_after_sukun_correction = []
    list_of_expected_letters_after_sukun_correction = []

    if len(list_of_actual_letters_before_sukun_correction) != len(list_of_expected_letters_before_sukun_correction):
        raise ValueError('bug appeared in "sukun_correction"')

    for each_actual_character, each_expected_letter in zip(list_of_actual_letters_before_sukun_correction,
                                                           list_of_expected_letters_before_sukun_correction):

        list_of_actual_letters_after_sukun_correction.append(remove_sukun_diacritics_if_exists(each_actual_character))
        list_of_expected_letters_after_sukun_correction.append(remove_sukun_diacritics_if_exists(each_expected_letter))

    return list_of_actual_letters_after_sukun_correction, list_of_expected_letters_after_sukun_correction


def remove_sukun_diacritics_if_exists(letter):
    normalized_char = unicodedata.normalize('NFC', letter)

    if u'ْ' in normalized_char:
        for c in normalized_char:
            if not unicodedata.combining(c):
                return unicodedata.normalize('NFC', c)
    else:
        return letter


def get_chars_count_for_each_word_in_current_sentence(sentence):

    count = 0
    chars_count_of_each_word = []
    for each_word in sentence:
        for each_char in each_word:
            if not unicodedata.combining(each_char):
                count += 1
        chars_count_of_each_word.append(count)
        count = 0

    return chars_count_of_each_word


def get_location_of_each_character_in_current_sentence(__list_of_actual_letters, __chars_count_for_each_word_in_current_sentence):

    list_of_actual_letters_with_its_location = []
    i = 0

    sum = 0
    for each_number in __chars_count_for_each_word_in_current_sentence:
        sum += each_number

    if sum != len(__list_of_actual_letters):
        raise ValueError('bug appeared in "get_location_of_each_character_in_current_sentence"')

    for count_of_letters in __chars_count_for_each_word_in_current_sentence:
        letter_position_object = LetterPosition()
        for x in range(0, count_of_letters):
            if count_of_letters == 1:
                letter_position_object.letter = __list_of_actual_letters[i]
                letter_position_object.location = 'firstOneLetter'
                list_of_actual_letters_with_its_location.append(deepcopy(letter_position_object))

            else:
                if x == 0:
                    letter_position_object.letter = __list_of_actual_letters[i]
                    letter_position_object.location = 'first'
                    list_of_actual_letters_with_its_location.append(deepcopy(letter_position_object))

                elif x == (count_of_letters - 1):
                    letter_position_object.letter = __list_of_actual_letters[i]
                    letter_position_object.location = 'last'
                    list_of_actual_letters_with_its_location.append(deepcopy(letter_position_object))

                else:
                    letter_position_object.letter = __list_of_actual_letters[i]
                    letter_position_object.location = 'middle'
                    list_of_actual_letters_with_its_location.append(deepcopy(letter_position_object))

            i += 1

    return list_of_actual_letters_with_its_location


def reform_word_after_sukun_correction(list_of_chars_with_its_position):
    list_of_words = []
    word = ""
    for each_char_object in list_of_chars_with_its_position:
        if each_char_object.location == 'firstOneLetter':
            list_of_words.append(each_char_object.letter)
        elif each_char_object.location != 'last':
            word += each_char_object.letter
        elif each_char_object.location == 'last':
            word += each_char_object.letter
            list_of_words.append(word)
            word = ""

    return list_of_words


def get_all_un_words_of_this_sentence_from_db(sentence_number):
    #undiacritized_word_in_selected_sentence_query = "select UnDiacritizedWord from parseddocument where LetterType='testing' and SentenceNumber = " + \
    #                          str(4591 + 414)

    undiacritized_word_in_selected_sentence_query = "select Word from listofwordsandsentencesineachdoc where SentenceNumber = " + \
                              str(sentence_number)

    cur.execute(undiacritized_word_in_selected_sentence_query)

    undiacritized_word_in_selected_sentence = cur.fetchall()
    # current_sentence = sorted(set(current_sentence), key=lambda x: current_sentence.index(x))
    undiacritized_word_in_selected_sentence = [eachTuple[0] for eachTuple in undiacritized_word_in_selected_sentence]
    #undiacritized_word_in_selected_sentence = [x[0] for x in groupby(undiacritized_word_in_selected_sentence)]

    list_of_un_diacritized_word = []
    for each_word in undiacritized_word_in_selected_sentence:
        nfkd_form = unicodedata.normalize('NFKD', each_word)
        unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٔ' or c == u'ٕ'])
        list_of_un_diacritized_word.append(unDiacritizedWord)

    return list_of_un_diacritized_word


def get_diac_version_with_smallest_dist(list_of_corrected_diacritized_words, list_of_undiacritized_words):

    list_of_actual_words_after_dictionary_correction = []

    if len(list_of_undiacritized_words) != len(list_of_corrected_diacritized_words):
        raise ValueError('bug appeared in "get_diac_version_with_smallest_dist"')

    for each_un_diacritized_word, each_corrected_word in zip(list_of_undiacritized_words, list_of_corrected_diacritized_words):

        minimum_error = 100000000
        dictionary_diacritized_words = get_corresponding_diacritized_versions(each_un_diacritized_word)

        if len(dictionary_diacritized_words) == 0:
            dictionary_diacritized_words.append(each_corrected_word)

        dictionary_diacritized_words_after_sukun_correction = sukun_correction_for_dictionary_words(dictionary_diacritized_words)

        if do_we_need_to_search_in_dictionary(dictionary_diacritized_words_after_sukun_correction, each_corrected_word):

            error_count = 0

            for each_word in dictionary_diacritized_words_after_sukun_correction:

                decomposed_dic_word, decomposed_act_word = decompose_word_into_letters(each_word, each_corrected_word)
                norm_dic_word, norm_act_word = normalize_words_under_comparison(decomposed_dic_word, decomposed_act_word)

                for each_diacritized_version_letter, each_current_word_letter in zip(norm_dic_word, norm_act_word):

                    if (len(each_diacritized_version_letter) - len(each_current_word_letter) == 1) or (
                            (len(each_diacritized_version_letter) - len(each_current_word_letter) == -1)):
                        error_count += 1

                    elif (len(each_diacritized_version_letter) - len(each_current_word_letter) == 2) or (
                            (len(each_diacritized_version_letter) - len(each_current_word_letter) == -2)):
                        error_count += 2

                    else:
                        for each_item_in_diacritized_version, each_item_in_current_word in zip(each_diacritized_version_letter, each_current_word_letter):
                            if each_item_in_diacritized_version != each_item_in_current_word:
                                error_count += 1

                if error_count < minimum_error:
                    minimum_error = error_count
                    selected_dictionary_word = each_word

            list_of_actual_words_after_dictionary_correction.append(selected_dictionary_word)
        else:
            list_of_actual_words_after_dictionary_correction.append(each_corrected_word)

    x = convert_list_of_words_to_list_of_chars(list_of_actual_words_after_dictionary_correction)
    return convert_list_of_words_to_list_of_chars(list_of_actual_words_after_dictionary_correction)


def get_corresponding_diacritized_versions(word):
    connect_to_db()

    selected_sentence_query = "select DiacritizedWord from dictionary where  UnDiacritizedWord = " + "'" +word + "'"
    cur.execute(selected_sentence_query)
    corresponding_diacritized_words = cur.fetchall()
    corresponding_diacritized_words = [each_word[0] for each_word in corresponding_diacritized_words]


    return corresponding_diacritized_words


def sukun_correction_for_dictionary_words(dictionary_list):
    dictionary_words_without_sukun = []
    overall = ""
    for each_word in dictionary_list:
        for each_char in each_word:
            spaChar = unicodedata.normalize('NFC', each_char)
            if u'ْ' in spaChar:
                    if not unicodedata.combining(spaChar):
                        overall += spaChar
                        dictionary_words_without_sukun.append(unicodedata.normalize('NFC', overall))
            else:
                overall += spaChar
        dictionary_words_without_sukun.append(unicodedata.normalize('NFC', overall))
        overall = ""
    return dictionary_words_without_sukun


def do_we_need_to_search_in_dictionary(dictionary, word):

    for each_word in dictionary:
        decomposed_dict, decomposed_act = decompose_word_into_letters(each_word, word)
        norm_dict, norm_act = normalize_words_under_comparison(decomposed_dict, decomposed_act)

        if len(norm_dict) != len(norm_act):
            raise ValueError("Bug Found In 'do_we_need_to_search_in_dictionary'")

        if sorted(norm_dict) == sorted(norm_act):
            return False

    for each_word in dictionary:
        decomposed_dict, decomposed_act = decompose_word_into_letters(each_word, word)
        norm_dict, norm_act = normalize_words_under_comparison(decomposed_dict, decomposed_act)
        for x in range(0, len(norm_act)):
            # compare letters before last letter
            if x < (len(norm_act) - 1):
                if norm_dict[x] != norm_act[x]:
                    # so diff is in first or middle letters
                    break
            else:
                # so diff is in last letter so ignore it
                return False

    return True


def decompose_word_into_letters(word_in_dictionary, actual_word):
    decomposed_dictionary_word = []
    decomposed_actual_word = []
    inter_med_list = []
    found_flag = False
    for each_letter in word_in_dictionary:
        if not unicodedata.combining(each_letter):
            if found_flag:
                decomposed_dictionary_word.append(inter_med_list)
            inter_med_list = []
            inter_med_list.append(each_letter)
            found_flag = True

        elif found_flag:
            inter_med_list.append(each_letter)
    # required because last character will not be added above, but here
    decomposed_dictionary_word.append(inter_med_list)

    inter_med_list = []
    found_flag = False
    for each_letter in actual_word:
        if not unicodedata.combining(each_letter):
            if found_flag:
                decomposed_actual_word.append(inter_med_list)
            inter_med_list = []
            inter_med_list.append(each_letter)
            found_flag = True

        elif found_flag:
            inter_med_list.append(each_letter)
            # required because last character will not be added above, but here
    decomposed_actual_word.append(inter_med_list)

    return decomposed_dictionary_word, decomposed_actual_word


def normalize_words_under_comparison(word_in_dictionary, actual_word):

    locale.setlocale(locale.LC_ALL, "")
    for x in range(0, len(word_in_dictionary)):
        word_in_dictionary[x].sort(cmp=locale.strcoll)

    locale.setlocale(locale.LC_ALL, "")
    for x in range(0, len(actual_word)):
        actual_word[x].sort(cmp=locale.strcoll)
    return word_in_dictionary, actual_word


def convert_list_of_words_to_list_of_chars(list_of_words):

    found_flag = False
    overall = ""
    comp = ""
    final_list_of_actual_letters_after_post_processing = []
    for each_word in list_of_words:
        for each_letter in each_word:
            if not unicodedata.combining(each_letter):
                if found_flag:
                    final_list_of_actual_letters_after_post_processing.append(comp)

                overall = each_letter
                found_flag = True
                comp = unicodedata.normalize('NFC', overall)
            elif found_flag:
                overall += each_letter
                comp = unicodedata.normalize('NFC', overall)

    final_list_of_actual_letters_after_post_processing.append(comp)

    return final_list_of_actual_letters_after_post_processing


def get_diacritization_error(actual_letters, expected_letters, sentence):
    list_of_object_error = []
    global total_error
    global total_chars_including_un_diacritized_target_letter
    global total_chars_not_including_un_diacritized_target_letter

    number_of_diacritization_errors = 0
    letter_location = 0

    if len(actual_letters) != len(expected_letters):
        raise ValueError('bug appeared in "get_diacritization_error"')

    for actual_letter, expected_letter in zip(actual_letters, expected_letters):
        error_object = ErrorDetails()
        decomposed_expected_letter = decompose_letter_into_chars_and_diacritics(expected_letter)
        letter_location += 1
        total_chars_including_un_diacritized_target_letter += 1
        if len(decomposed_expected_letter) > 1:
            total_chars_not_including_un_diacritized_target_letter += 1
            if actual_letter != expected_letter:
                error_object.actual_letter = actual_letter
                error_object.expected_letter = expected_letter
                error_object.error_location = letter_location
                error_object.word = get_word_that_has_error(letter_location, sentence)

                list_of_object_error.append(deepcopy(error_object))
                number_of_diacritization_errors += 1

    total_error += number_of_diacritization_errors

    print 'total error in this sentence', number_of_diacritization_errors
    print 'total error in all sentences: ', total_error

    return list_of_object_error


def get_diacritization_error_without_counting_last_letter(actual_letters, expected_letters, sentence):

    list_of_object_error = []
    global total_error_without_last_letter

    number_of_diacritization_errors = 0
    letter_location = 0
    error_locations = []

    if len(actual_letters) != len(expected_letters):
        raise ValueError('bug appeared in "get_diacritization_error_without_counting_last_letter"')

    for actual_letter, expected_letter in zip(actual_letters, expected_letters):
        error_object = ErrorDetails()
        letter_location += 1
        if actual_letter.location != 'last' and expected_letter.location != 'last':
            decomposed_expected_letter = decompose_letter_into_chars_and_diacritics(expected_letter.letter)
            if actual_letter.location == expected_letter.location:

                if len(decomposed_expected_letter) > 1:
                    if actual_letter.letter != expected_letter.letter:
                        error_object.actual_letter = actual_letter.letter
                        error_object.expected_letter = expected_letter.letter
                        error_object.error_location = letter_location
                        error_object.word = get_word_that_has_error(letter_location, sentence)

                        list_of_object_error.append(deepcopy(error_object))
                        number_of_diacritization_errors += 1
            else:
                raise ValueError('bug appeared in "get_diacritization_error_without_counting_last_letter"')

    total_error_without_last_letter += number_of_diacritization_errors

    print 'total error in this sentence (without Last Letter):', number_of_diacritization_errors
    print 'total error in all sentences (without Last Letter):', total_error_without_last_letter

    return list_of_object_error


def decompose_letter_into_chars_and_diacritics(expected_letter):
    decomposed_letter = []
    for each_letter in expected_letter:
        decomposed_letter.append(each_letter)

    return decomposed_letter


def get_word_that_has_error(error_location, sentence):
    counter = 0
    for each_word in sentence:
        each_word = unicodedata.normalize('NFD', each_word)
        for each_char in each_word:
            if not unicodedata.combining(each_char):
                counter += 1
                if error_location == counter:
                    return each_word


def write_data_into_excel_file(errors, current_sentence, diacritization_error_excel_file_path, current_row_in_excel_file):
    wb = open_workbook(diacritization_error_excel_file_path)
    w = copy(wb)
    worksheet = w.get_sheet(0)

    # global current_row_in_excel_file
    current_row_in_excel_file += 1
    column = 0

    for each_object in errors:

        worksheet.write(current_row_in_excel_file, column, each_object.actual_letter)

        column = 1
        worksheet.write(current_row_in_excel_file, column, each_object.expected_letter)

        column = 2
        worksheet.write(current_row_in_excel_file, column, each_object.error_location)

        column = 3
        worksheet.write(current_row_in_excel_file, column, each_object.word)

        current_row_in_excel_file += 1
        column = 0

    all_sentence = ''
    for each_word in current_sentence:
        all_sentence += each_word + ' '

    worksheet.write(current_row_in_excel_file, column, all_sentence)

    current_row_in_excel_file += 1

    w.save(diacritization_error_excel_file_path)
    workbook.close()

    return current_row_in_excel_file


if __name__ == "__main__":
    connect_to_db()
    list_of_sentence_numbers = get_list_of_sentence_numbers()

    os.chdir(path)
    result = [i for i in glob.glob('*.{}'.format(extension))]
    current_sentence_counter = 0

    for file_name in result:
        selected_sentence, sentence_number = get_sentence_from_db(current_sentence_counter, list_of_sentence_numbers)

        rnn_output_for_one_seq = read_csv_file_of_a_predicted_sentence(file_name)

        neurons_locations_with_highest_output_for_a_seq = \
            get_neurons_numbers_with_highest_output_value(rnn_output_for_one_seq)

        connect_to_db()

        list_of_diacritized_letters = get_all_letters_from_db()

        actual_letters_before_sukun_correction = get_actual_letters(list_of_diacritized_letters,
                                                                    neurons_locations_with_highest_output_for_a_seq)

        expected_letters_before_sukun_correction = get_expected_letters(sentence_number)

        actual_letters_after_sukun_correction, expected_letters_after_sukun_correction = \
            sukun_correction(actual_letters_before_sukun_correction, expected_letters_before_sukun_correction)

        chars_count_for_each_word_in_current_sentence = get_chars_count_for_each_word_in_current_sentence(selected_sentence)

        location_of_each_char = get_location_of_each_character_in_current_sentence(actual_letters_after_sukun_correction, chars_count_for_each_word_in_current_sentence)

        list_of_words_in_sent_after_sukun_correction = reform_word_after_sukun_correction(location_of_each_char)

        connect_to_db()

        list_of_undiacritized_words_in_current_sentence = get_all_un_words_of_this_sentence_from_db(sentence_number)

        list_of_chars_in_sent_after_sukun_and_dict_correction = get_diac_version_with_smallest_dist(list_of_words_in_sent_after_sukun_correction, list_of_undiacritized_words_in_current_sentence)

        # to get character position
        location_of_each_char_for_actual_op = \
            get_location_of_each_character_in_current_sentence(list_of_chars_in_sent_after_sukun_and_dict_correction,
                                                               chars_count_for_each_word_in_current_sentence)

        location_of_each_char_for_expected_op = \
            get_location_of_each_character_in_current_sentence(expected_letters_after_sukun_correction,
                                                               chars_count_for_each_word_in_current_sentence)
        # end of get character position

        # DER_Testing Calculation
        list_of_error = get_diacritization_error(list_of_chars_in_sent_after_sukun_and_dict_correction,
                                                 expected_letters_after_sukun_correction,
                                                 selected_sentence)

        list_of_error_without_counting_last_letter = get_diacritization_error_without_counting_last_letter(
                                                        location_of_each_char_for_actual_op,
                                                        location_of_each_char_for_expected_op,
                                                        selected_sentence)
        # End DER_Testing Calculation

        excel1 = current_row_excel_1
        current_row_excel_1 = write_data_into_excel_file(list_of_error, selected_sentence,
                                                         diacritization_error_excel_file_path, excel1)

        excel2 = current_row_excel_2
        current_row_excel_2 = write_data_into_excel_file(list_of_error_without_counting_last_letter,
                                                         selected_sentence,
                                                         diacritization_error_without_last_letter_excel_file_path,
                                                         excel2)

        current_sentence_counter += 1
        print 'sentence number: ', current_sentence_counter

    print "Total Chars Not Including Undiacritized Target Letter: ", total_chars_not_including_un_diacritized_target_letter
    print "Total Chars Including Undiacritized Target Letter: ", total_chars_including_un_diacritized_target_letter
