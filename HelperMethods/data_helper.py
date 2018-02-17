import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime
import matplotlib
from collections import Counter
from itertools import chain


def establish_db_connection():
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
    establish_db_connection()
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


def load_cnn_blstm_table(data_type):
    start_time = datetime.datetime.now()

    query = "select UnDiacritizedCharacter, Diacritics, SentenceNumber, location from ParsedDocument where LetterType="\
            + "'%s'" % data_type

    cur.execute(query)

    data = cur.fetchall()
    data = np.array(data)

    end_time = datetime.datetime.now()
    print("load_cnn_blstm_table takes : ", end_time - start_time)

    return data


def get_input_table():
    start_time = datetime.datetime.now()
    establish_db_connection()
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
    establish_db_connection()
    query = "select * from diacritics_and_undiacritized_letter_one_hot_encoding"
    cur.execute(query)

    labels_and_equiv_encoding = cur.fetchall()
    labels_and_equiv_encoding = np.array(labels_and_equiv_encoding)

    end_time = datetime.datetime.now()
    print("get_db_label_table takes : ", end_time - start_time)
    cur.close()
    return labels_and_equiv_encoding


def load_nn_input_dataset_numpy(data_table):
    start_time = datetime.datetime.now()
    nn_input = []

    input_and_equiv_encoding = get_input_table()

    for each_row in data_table:
        raw_input_data = each_row[0]
        location = each_row[1]

        index_of_raw_input_data = np.where(input_and_equiv_encoding == raw_input_data)

        if np.size(index_of_raw_input_data) != 0:
            one_hot_encoding = input_and_equiv_encoding[np.min(index_of_raw_input_data[0]), 1]

            if location == 'first':
                one_hot_encoding = one_hot_encoding + '100'
            elif location == 'middle':
                one_hot_encoding = one_hot_encoding + '010'
            else:
                one_hot_encoding = one_hot_encoding + '001'

            nn_input.append(list(map(int, one_hot_encoding)))

    nn_input_np_array = np.array(nn_input)

    nn_input_np_array = nn_input_np_array.astype(np.float)

    end_time = datetime.datetime.now()

    print("load_nn_input_dataset_numpy takes : ", end_time - start_time)

    return nn_input_np_array


def load_nn_input_dataset_string(data_table):
    start_time = datetime.datetime.now()
    nn_input = []

    input_and_equiv_encoding = get_input_table()

    for each_row in data_table:
        raw_input_data = each_row[0]
        location = each_row[1]

        index_of_raw_input_data = np.where(input_and_equiv_encoding == raw_input_data)

        if np.size(index_of_raw_input_data) != 0:
            one_hot_encoding = input_and_equiv_encoding[np.min(index_of_raw_input_data[0]), 1]

            if location == 'first':
                one_hot_encoding = one_hot_encoding + '100'
            elif location == 'middle':
                one_hot_encoding = one_hot_encoding + '010'
            else:
                one_hot_encoding = one_hot_encoding + '001'

            nn_input.append(one_hot_encoding)

    end_time = datetime.datetime.now()

    print("load_nn_input_dataset_string takes : ", end_time - start_time)

    return np.array(nn_input)


def load_nn_labels_dataset_numpy(data_table):
    start_time = datetime.datetime.now()
    nn_labels = []

    labels_and_equiv_encoding = get_label_table()

    for each_row in data_table:
        if each_row[1] != '':
            raw_input_data = each_row[1]
        else:
            raw_input_data = each_row[0]

        index_of_raw_label_data = np.where(labels_and_equiv_encoding == raw_input_data)

        if np.size(index_of_raw_label_data) != 0:
            label = np.min(index_of_raw_label_data[0])
            nn_labels.append(label)

    nn_label_np_array = np.array(nn_labels)

    nn_label_np_array = nn_label_np_array.astype(np.float)

    end_time = datetime.datetime.now()

    print("load_nn_labels_dataset_numpy takes : ", end_time - start_time)

    return nn_label_np_array


def load_nn_labels_dataset_string(data_table):
    start_time = datetime.datetime.now()
    nn_labels = []

    labels_and_equiv_encoding = get_label_table()

    for each_row in data_table:
        if each_row[1] != '':
            raw_input_data = each_row[1]
        else:
            raw_input_data = each_row[0]

        index_of_raw_label_data = np.where(labels_and_equiv_encoding == raw_input_data)

        if np.size(index_of_raw_label_data) != 0:
            label = labels_and_equiv_encoding[index_of_raw_label_data[0], 2][0]
            label = label.replace('\n', "")
            label = list(map(int, label))

            nn_labels.append(label)

    end_time = datetime.datetime.now()

    print("load_nn_labels_dataset_string takes : ", end_time - start_time)

    return np.array(nn_labels)


def load_nn_seq_lengths(data_table):

    start_time = datetime.datetime.now()

    sent_num, sen_len = np.unique(np.hstack(data_table), return_counts=True)

    end_time = datetime.datetime.now()

    print("load_nn_seq_lengths takes : ", end_time - start_time)

    return sent_num, sen_len


def pad_sentences(x, sent_len, req_char_index, window_size):
    start_time = datetime.datetime.now()

    padded_sent = []
    start_range = 0
    end_range = 0

    for each_sent in range(0, len(sent_len)):

        end_range = sent_len[each_sent] + end_range
        extracted_sent = x[start_range: end_range]
        padded_sent.append(padding(extracted_sent, req_char_index, window_size))
        start_range = end_range

    end_time = datetime.datetime.now()

    vocabulary, vocabulary_inv = build_vocab(padded_sent)

    padded_sent = build_input_data(padded_sent, vocabulary)

    print("pad_sentences takes : ", end_time - start_time)

    return padded_sent, vocabulary, vocabulary_inv


def padding(extracted_sent, req_char_index, window_size):

    padded_sent = []

    for index in range(0, len(extracted_sent)):
        new_list = ['pad'] * window_size
        new_list[req_char_index] = extracted_sent[index]

        # before req index
        end_range = index - 1

        start_range = index - req_char_index
        if start_range < 0:
            start_range = 0

        if start_range >= 0 and end_range >= 0:
            extracted_chars_in_cert_range = extracted_sent[start_range: (end_range + 1)]
            num_of_elem = np.size(extracted_chars_in_cert_range)
            new_list[(req_char_index - num_of_elem): req_char_index] = extracted_chars_in_cert_range

        # after req index
        start_range = index + 1

        end_range = index + req_char_index + 1
        if end_range >= np.size(extracted_sent):
            end_range = np.size(extracted_sent) - 1

        if start_range <= (np.size(extracted_sent) - 1) and end_range <= (np.size(extracted_sent) - 1):
            extracted_chars_in_cert_range = extracted_sent[start_range: (end_range + 1)]
            num_of_elem = np.size(extracted_chars_in_cert_range)
            new_list[(req_char_index + 1): (req_char_index + num_of_elem + 1)] = extracted_chars_in_cert_range

        padded_sent.append(new_list)

    return padded_sent


def build_vocab(sentences):
    start_time = datetime.datetime.now()
    all_chars = list(matplotlib.cbook.flatten(sentences))
    chars_counts = Counter(all_chars)

    # Mapping from index to char
    vocabulary_inv = [x[0] for x in chars_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from char to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    end_time = datetime.datetime.now()

    print("build_vocab takes : ", end_time - start_time)

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    sentences = list(chain(*sentences))
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


if __name__ == "__main__":

    establish_db_connection()
    dataset = load_dataset_by_type('training')
    load_nn_input_dataset_numpy(dataset[:, [0, 8]])
    load_nn_labels_dataset_numpy(dataset[:, [0, 1]])
    load_nn_seq_lengths(dataset[:, [3]])
