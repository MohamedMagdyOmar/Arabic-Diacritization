# -*- coding: utf-8 -*-
import unicodedata
import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime


class Repository:

    def __init__(self):
        self.db = MySQLdb.connect(host="127.0.0.1",  # your host, usually localhost
                                  user="root",  # your username
                                  passwd="Islammega88",  # your password
                                  db="mstdb",  # name of the data base
                                  cursorclass=MySQLdb.cursors.SSCursor,
                                  use_unicode=True,
                                  charset="utf8",
                                  init_command='SET NAMES UTF8')

        self.cur = self.db.cursor()
        self.query_result = ""
        self.query = ""
        self.list_of_un_diacritized_word = []

    def get_input_data(self):
        start_time = datetime.datetime.now()

        self.query = "select UnDiacritizedCharacter, UnDiacritizedCharacterOneHotEncoding from UnDiacOneHotEncoding"
        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("get label data takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result

    def get_label_data__separate_letter_diacritics(self):
        """
        use this method if you want to predict diacritics, and letters separately
        """
        start_time = datetime.datetime.now()

        self.query = "select * from diacritics_and_undiacritized_letter_one_hot_encoding"
        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("get label data takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result

    def get_label_data__letter_combination_with_diacritics(self):
        """
        use this method if you want to predict each letter with all possible combinations of diacritics(480,
        Not recommended)
        """
        start_time = datetime.datetime.now()
        self.query = "select * from diaconehotencoding"
        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("get label data takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result

    def get_label_data_diacritics_only(self):
        start_time = datetime.datetime.now()

        self.query = "select * from distinctdiacritics"
        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("get label data takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result

    def get_dataset_sentence_numbers_using_dataset_type(self, sentence_type):
        """
        old name: get_list_of_sentence_numbers_by
        sentence_type: string :training, testing
        """

        self.query = "select distinct SentenceNumber from parseddocument where LetterType = " + \
                     "'" + sentence_type + "'" + " order by SentenceNumber asc, idCharacterNumber asc "

        self.cur.execute(self.query)

        self.query_result = (self.cur.fetchall())
        self.query_result = [each_number[0] for each_number in self.query_result]
        self.query_result = list(map(int, self.query_result))
        self.cur.close()
        return self.query_result

    def get_dataset_words_of_a_sentence_by_sentence_number_and_word_type(self, sentence_number, category):
        """
        old name: get_sentence_by
        sentence_number: int
        category: string : training, testing
        """
        self.query = "select Word from listofwordsandsentencesineachdoc where SentenceNumber = " + \
                     str(sentence_number) + " and wordtype=" + "'" + category + "'"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = [eachTuple[0] for eachTuple in self.query_result]
        self.cur.close()
        return self.query_result

    def get_dataset_sentence_info(self, sentence_number, category):
        """
        old name: get_chars_by_sentence
        sentence_number: int
        category: string: : training, testing
        """
        self.query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, " \
                     "DiacritizedCharacter, location, UnDiacritizedWord from ParsedDocument where LetterType=" + \
                     "'%s'" % category + " and SentenceNumber=" + "'%i'" % str(sentence_number) + "order by " \
                     "SentenceNumber asc, idCharacterNumber asc"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        self.cur.close()
        return self.query_result

    def get_dataset_words_and_sentence_number_by_category(self, category):
        """
        old name: get_all_sentences_by
        category: string: : training, testing
        """
        self.query = "select Word, SentenceNumber from listofwordsandsentencesineachdoc where wordtype=" + "'" + \
                     category + "'"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()

        self.cur.close()
        return np.array(self.query_result)

    def get_dataset_all_words_and_sentence_number(self):
        """
        old name: get_all_sentences_by
        category: string: : training, testing
        """
        self.query = "select * from listofwordsandsentencesineachdoc"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()

        self.cur.close()

        return np.array(self.query_result)

    def get_all_diacritics(self):
        self.query = "select diacritics from arabic_diacritics"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = [eachTuple[0] for eachTuple in self.query_result]
        self.cur.close()
        return self.query_result

    def get_dataset_un_diacritized_chars_by_sentence_number_category(self, sentence_number, category):
        """
        old name: get_un_diacritized_chars_by
        sentence_number: int
        category: string: : training, testing
        """
        self.query = "select UnDiacritizedCharacter from parseddocument where " \
                     "LetterType = " + "'" + category + "'" + "and SentenceNumber = " + str(sentence_number)

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = [eachTuple[0] for eachTuple in self.query_result]
        self.cur.close()
        return self.query_result

    def get_dataset_diacritized_chars_by_sentence_number_category(self, sentence_number, category):
        """
        old name: get_diacritized_chars_by
        sentence_number: int
        category: string: : training, testing
        """
        self.query = "select DiacritizedCharacter from parseddocument where " \
                     "LetterType = " + "'" + category + "'" + "and SentenceNumber = " + str(sentence_number)

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = [eachTuple[0] for eachTuple in self.query_result]
        self.cur.close()

        return self.query_result

    def get_diacritized_chars(self):
        """
        old name: get_available_diacritized_chars
        """
        self.query = "select DiacritizedCharacter from diaconehotencoding"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = [eachTuple[0] for eachTuple in self.query_result]
        self.cur.close()
        return self.query_result

    def get_dataset_un_diacritized_words_using_sentence_number(self, sentence_number):
        """
        old name: get_available_diacritics_and_un_diacritized_chars
        sentence_number: int
        """
        self.query = "select word from listofwordsandsentencesineachdoc where SentenceNumber = " + str(sentence_number)

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()

        self.query_result = [eachTuple[0] for eachTuple in self.query_result]

        for each_word in self.query_result:
            nfkd_form = unicodedata.normalize('NFKD', each_word)
            undiacritized_word = u"".join(
                [c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٓ' or c == u'ٔ' or c == u'ٕ'])
            self.list_of_un_diacritized_word.append(undiacritized_word)

        self.cur.close()
        return self.list_of_un_diacritized_word

    def get_dataset_un_diacritized_words_and_sentence_number(self, category):
        """
        old name: get_dataset_un_diacritized_words_and_sentence_number
        """
        self.query = "select word, SentenceNumber from listofwordsandsentencesineachdoc where wordtype=" + "'" + \
                     category + "'"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()

        self.query_result = list(map(list, self.query_result))

        for each_word in self.query_result:
            nfkd_form = unicodedata.normalize('NFKD', each_word[0])
            each_word[0] = u"".join(
                [c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٓ' or c == u'ٔ' or c == u'ٕ'])

        self.cur.close()

        return np.array(self.query_result)

    def get_all_diacritized_versions_using_undiacritized_word(self, un_diacritized_word):
        """
        old name: get_dictionary_all_diacritized_version_of
        """
        self.query = "select DiacritizedWord from dictionary where  UnDiacritizedWord = " + "'" + un_diacritized_word \
                     + "'"

        self.cur.execute(self.query)
        self.query_result = self.cur.fetchall()
        self.query_result = [each_word[0] for each_word in self.query_result]
        self.cur.close()
        return self.query_result

    def get_dictionary(self):
        self.query = "select DiacritizedWord, UnDiacritizedWord from dictionary"

        self.cur.execute(self.query)
        self.query_result = self.cur.fetchall()

        self.cur.close()
        return np.array(self.query_result)

    def get_data_set(self):
        """
        old name: load_data_set
        """
        start_time = datetime.datetime.now()

        self.query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, " \
                     "DiacritizedCharacter, location from ParsedDocument order by SentenceNumber asc, " \
                     "idCharacterNumber asc "

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("load_data_set takes : ", end_time - start_time)

        self.cur.close()
        return self.query_result

    def get_dataset_by_type(self, category):
        """
        old name: load_dataset_by_type
        """
        start_time = datetime.datetime.now()

        self.query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, " \
                     "DiacritizedCharacter,location, UnDiacritizedWord from ParsedDocument where LetterType=" + \
                     "'%s'" % category + " order by SentenceNumber asc, idCharacterNumber asc"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("load_dataset_by_type takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result

    def get_dataset_by_type_and_sentence_number_for_testing_purpose(self, category, sentence_number):
        """
        old name: load_dataset_by_type_and_sentence_number_for_testing_purpose
        """
        start_time = datetime.datetime.now()

        self.query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, " \
                     "DiacritizedCharacter, location, UnDiacritizedWord from ParsedDocument where LetterType=" + \
                     "'%s'" % category + " and SentenceNumber=" + "'%s'" % sentence_number + \
                     " order by SentenceNumber asc, idCharacterNumber asc limit 1000"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("load_dataset_by_type takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result

    def get_testing_dataset(self):
        """
        old name: load_testing_dataset
        """
        start_time = datetime.datetime.now()

        self.query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word, " \
                     "DiacritizedCharacter, location, UnDiacritizedWord from ParsedDocument where LetterType=" + \
                     "'%s'" % "testing"

        self.cur.execute(self.query)

        self.query_result = self.cur.fetchall()
        self.query_result = np.array(self.query_result)

        end_time = datetime.datetime.now()
        print("load_testing_dataset takes : ", end_time - start_time)
        self.cur.close()
        return self.query_result
