# -*- coding: utf-8 -*-
import unicodedata
import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime


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


def get_list_of_sentence_numbers_by(sentence_type):

    connect_to_db()
    get_sentence_number_query = "select distinct SentenceNumber from parseddocument where LetterType = " + \
                                "'" + sentence_type + "'" + " order by idCharacterNumber asc "

    cur.execute(get_sentence_number_query)

    sentence_numbers = (cur.fetchall())
    sentence_numbers = [each_number[0] for each_number in sentence_numbers]
    sentence_numbers = list(map(int, sentence_numbers))
    cur.close()
    return sentence_numbers


def get_sentence_by(sentence_number):

    connect_to_db()
    get_sentence_query = "select Word from listofwordsandsentencesineachdoc where SentenceNumber = " + \
                         str(sentence_number)

    cur.execute(get_sentence_query)

    current_sentence = cur.fetchall()
    current_sentence = [eachTuple[0] for eachTuple in current_sentence]
    cur.close()
    return current_sentence


def get_all_diacritics():

    connect_to_db()
    get_all_diacritics_query = "select diacritics from arabic_diacritics"

    cur.execute(get_all_diacritics_query)

    distinct_diacritics = cur.fetchall()
    distinct_diacritics = [eachTuple[0] for eachTuple in distinct_diacritics]
    cur.close()
    return distinct_diacritics


def get_un_diacritized_chars_by(sentence_number, sentence_type):

    connect_to_db()
    get_un_diacritized_chars_query = "select UnDiacritizedCharacter from parseddocument where " \
                                     "LetterType = " + "'" + sentence_type + "'" + "and SentenceNumber = " + str(sentence_number)

    cur.execute(get_un_diacritized_chars_query)

    un_diacritized_chars = cur.fetchall()
    un_diacritized_chars = [eachTuple[0] for eachTuple in un_diacritized_chars]
    cur.close()
    return un_diacritized_chars


def get_diacritized_chars_by(sentence_number, sentence_type):

    connect_to_db()
    get_diacritized_chars_query = "select DiacritizedCharacter from parseddocument where " \
                                  "LetterType = " + "'" + sentence_type + "'" + "and SentenceNumber = " + str(sentence_number)

    cur.execute(get_diacritized_chars_query)

    diacritized_chars = cur.fetchall()
    diacritized_chars = [eachTuple[0] for eachTuple in diacritized_chars]
    cur.close()
    return diacritized_chars


def get_available_diacritized_chars():

    connect_to_db()
    get_available_diacritized_chars_query = "select DiacritizedCharacter from diaconehotencoding"

    cur.execute(get_available_diacritized_chars_query)

    diacritized_chars = cur.fetchall()
    diacritized_chars = [eachTuple[0] for eachTuple in diacritized_chars]
    cur.close()
    return diacritized_chars


def get_available_diacritics_and_un_diacritized_chars():

    connect_to_db()
    get_available_diacritized_chars_query = "select label from diacritics_and_undiacritized_letter_one_hot_encoding"

    cur.execute(get_available_diacritized_chars_query)

    diacritized_chars = cur.fetchall()
    diacritized_chars = [eachTuple[0] for eachTuple in diacritized_chars]
    cur.close()
    return diacritized_chars


def get_un_diacritized_words_from(sentence_number, sentence_type):

    connect_to_db()
    get_un_diacritized_words_query = "select word from listofwordsandsentencesineachdoc " \
                                     "where SentenceNumber = " + str(sentence_number)

    cur.execute(get_un_diacritized_words_query)

    un_diacritized_words = cur.fetchall()

    un_diacritized_words = [eachTuple[0] for eachTuple in un_diacritized_words]

    list_of_un_diacritized_word = []
    for each_word in un_diacritized_words:
        nfkd_form = unicodedata.normalize('NFKD', each_word)
        undiacritized_word = u"".join([c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٓ' or c == u'ٔ' or c == u'ٕ'])
        list_of_un_diacritized_word.append(undiacritized_word)
    cur.close()
    return list_of_un_diacritized_word


def get_all_un_diacritized_words_in_sentences():

    connect_to_db()
    get_un_diacritized_words_query = "select word, SentenceNumber from listofwordsandsentencesineachdoc "

    cur.execute(get_un_diacritized_words_query)

    un_diacritized_words = cur.fetchall()

    un_diacritized_words = list(map(list, un_diacritized_words))

    for each_word in un_diacritized_words:
        nfkd_form = unicodedata.normalize('NFKD', each_word[0])
        each_word[0] = u"".join([c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٓ' or c == u'ٔ' or c == u'ٕ'])
    cur.close()

    return np.array(un_diacritized_words)


def get_dictionary_all_diacritized_version_of(un_diacritized_word):
    connect_to_db()
    selected_sentence_query = "select DiacritizedWord from dictionary where  UnDiacritizedWord = " + "'" \
                              + un_diacritized_word + "'"

    cur.execute(selected_sentence_query)
    corresponding_diacritized_words = cur.fetchall()
    corresponding_diacritized_words = [each_word[0] for each_word in corresponding_diacritized_words]
    cur.close()
    return corresponding_diacritized_words


def get_dictionary():
    connect_to_db()
    selected_sentence_query = "select DiacritizedWord, UnDiacritizedWord from dictionary"

    cur.execute(selected_sentence_query)
    corresponding_diacritized_words = cur.fetchall()

    cur.close()
    return np.array(corresponding_diacritized_words)


def get_input_table():
    start_time = datetime.datetime.now()
    connect_to_db()
    query = "select UnDiacritizedCharacter, UnDiacritizedCharacterOneHotEncoding from UnDiacOneHotEncoding"
    cur.execute(query)

    input_and_equiv_encoding = cur.fetchall()
    input_and_equiv_encoding = np.array(input_and_equiv_encoding)

    end_time = datetime.datetime.now()
    print("get_db_input_table takes : ", end_time - start_time)
    cur.close()
    return input_and_equiv_encoding


def get_label_table():
    start_time = datetime.datetime.now()
    connect_to_db()
    query = "select * from diacritics_and_undiacritized_letter_one_hot_encoding"
    cur.execute(query)

    labels_and_equiv_encoding = cur.fetchall()
    labels_and_equiv_encoding = np.array(labels_and_equiv_encoding)

    end_time = datetime.datetime.now()
    print("get_db_label_table takes : ", end_time - start_time)
    cur.close()
    return labels_and_equiv_encoding


def load_data_set():
    start_time = datetime.datetime.now()

    query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, DiacritizedCharacter, " \
            "location from ParsedDocument order by SentenceNumber asc"

    cur.execute(query)

    data = cur.fetchall()
    data = np.array(data)

    end_time = datetime.datetime.now()
    print("load_data_set takes : ", end_time - start_time)

    cur.close()
    return data


def load_dataset_by_type(data_type):

    start_time = datetime.datetime.now()
    connect_to_db()
    query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, DiacritizedCharacter, " \
            "location, UnDiacritizedWord from ParsedDocument where LetterType=" + \
            "'%s'" % data_type + " order by SentenceNumber asc"

    cur.execute(query)

    data = cur.fetchall()
    data = np.array(data)

    end_time = datetime.datetime.now()
    print("load_dataset_by_type takes : ", end_time - start_time)
    cur.close()
    return data


def load_testing_dataset():

    start_time = datetime.datetime.now()
    connect_to_db()
    query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, DiacritizedCharacter, " \
            "location, UnDiacritizedWord from ParsedDocument where LetterType=" + \
            "'%s'" % "testing"

    cur.execute(query)

    data = cur.fetchall()
    data = np.array(data)

    end_time = datetime.datetime.now()
    print("load_testing_dataset takes : ", end_time - start_time)
    cur.close()
    return data

