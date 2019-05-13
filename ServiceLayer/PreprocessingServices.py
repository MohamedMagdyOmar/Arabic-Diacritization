import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime
import matplotlib
from collections import Counter
from itertools import chain
import unicodedata2
import numpy


class Services:

    def __init__(self):
        self.extracted_sent = ''
        self.padded_sentence = []
        self.start_time = ''
        self.end_time = ''

    def preprocess(self, x, char_position, window_size):
        """
        :param x: input data, 2d numpy array, first column is letter, second column is sentence number
        :param char_position: position of the character
        :param window_size:
        :old name: pad_sentences
        :return: padded sentence
        """
        self.start_time = datetime.datetime.now()

        num_of_sentences = numpy.unique(x[:, [1]], return_counts=True)

        for sent_num, char_count in zip(num_of_sentences[0], num_of_sentences[1]):
            self.extracted_sent = np.extract(x[:, [1]] == sent_num, x[:, [0]])
            self.padded_sentence.append(self.padding(char_position, window_size))

        self.end_time = datetime.datetime.now()

        print('Padding Sentence takes: ', self.end_time - self.start_time)

        return self.padded_sentence

    def padding(self, req_char_index_non_zero_index, window_size):

        padded_sent = []
        extracted_chars_in_cert_range = ""
        number_of_elements_before_target_char = req_char_index_non_zero_index - 1
        number_of_elements_after_target_char = window_size - req_char_index_non_zero_index
        num_of_elem = 0
        for index in range(0, len(self.extracted_sent)):
            new_list = ['pad'] * window_size
            new_list[req_char_index_non_zero_index - 1] = self.extracted_sent[index]

            # before req index
            end_range_for_extracted_sent = index - 1
            start_range_for_extracted_sent = index - number_of_elements_before_target_char

            if start_range_for_extracted_sent < 0:
                start_range_for_extracted_sent = 0

            if end_range_for_extracted_sent >= 0 and number_of_elements_before_target_char > 0:
                extracted_chars_in_cert_range = self.extracted_sent[
                                                start_range_for_extracted_sent: (end_range_for_extracted_sent + 1)]
                num_of_elem = np.size(extracted_chars_in_cert_range)

            end_range_for_new_list = req_char_index_non_zero_index - 1
            start_range_for_new_list = end_range_for_new_list - num_of_elem
            if start_range_for_new_list < 0:
                start_range_for_new_list = 0

            if end_range_for_new_list > 0 and num_of_elem > 0 and number_of_elements_before_target_char > 0:
                new_list[start_range_for_new_list: (end_range_for_new_list)] = extracted_chars_in_cert_range

            # after req index
            start_range_for_extracted_sent = index + 1
            if start_range_for_extracted_sent == len(self.extracted_sent):
                number_of_elements_after_target_char = 0

            end_range_for_extracted_sent = index + number_of_elements_after_target_char

            if number_of_elements_after_target_char > 0 and end_range_for_extracted_sent >= len(self.extracted_sent):
                extracted_chars_in_cert_range = self.extracted_sent[
                                                start_range_for_extracted_sent:]
                num_of_elem = np.size(extracted_chars_in_cert_range)

            elif number_of_elements_after_target_char > 0:
                extracted_chars_in_cert_range = self.extracted_sent[
                                                start_range_for_extracted_sent: (end_range_for_extracted_sent + 1)]
                num_of_elem = np.size(extracted_chars_in_cert_range)

            start_range_for_new_list = req_char_index_non_zero_index
            end_range_for_new_list = start_range_for_new_list + num_of_elem
            if end_range_for_new_list > window_size:
                end_range_for_new_list = window_size - 1

            if num_of_elem > 0 and number_of_elements_after_target_char > 0 and end_range_for_new_list == window_size:
                new_list[start_range_for_new_list:] = extracted_chars_in_cert_range

            elif num_of_elem > 0 and number_of_elements_after_target_char > 0:
                new_list[start_range_for_new_list: end_range_for_new_list] = extracted_chars_in_cert_range

            padded_sent.append(new_list)

        return padded_sent

    def build_vocab(self, padded_sentence):
        """
        note: old implementation takes "sentences" as an input
        :param padded_sentence: either a list, or any kind of list of list, anyway it will be flatted
        :return: [vocabulary, vocabulary_inv]
        """
        self.start_time = datetime.datetime.now()

        all_chars = list(matplotlib.cbook.flatten(padded_sentence))

        chars_counts = Counter(all_chars)

        # Mapping from index to char
        vocabulary_inv = [x[0] for x in chars_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))

        # Mapping from char to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        self.end_time = datetime.datetime.now()

        print("build_vocab takes : ", self.end_time - self.start_time)

        return [vocabulary, vocabulary_inv]

    def build_input_data(self, vocabulary):
        """
        convert "padded sentence" from letters, to corresponding numbers
        :param vocabulary: dictionary for available vocabulary
        :return: list of list for data set
        """
        self.padded_sentence = list(chain(*self.padded_sentence))
        x = np.array([[vocabulary[word] for word in sentence] for sentence in self.padded_sentence])
        return x

