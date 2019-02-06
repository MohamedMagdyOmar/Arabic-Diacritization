import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime


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


def load_dataset_table(data_type):

    start_time = datetime.datetime.now()

    query = "select UnDiacritizedCharacter, Diacritics, LetterType, SentenceNumber,Word " \
            ", DiacritizedCharacter, location from ParsedDocument where LetterType=" + \
            "'%s'" % data_type

    cur.execute(query)

    data = cur.fetchall()
    data = np.array(data)

    end_time = datetime.datetime.now()
    print("load_db_training_table takes : ", end_time - start_time)

    return data


def get_input_table():
    start_time = datetime.datetime.now()

    query = "select UnDiacritizedCharacter, UnDiacritizedCharacterOneHotEncoding from UnDiacOneHotEncoding"
    cur.execute(query)

    input_and_equiv_encoding = cur.fetchall()
    input_and_equiv_encoding = np.array(input_and_equiv_encoding)

    end_time = datetime.datetime.now()
    print("get_db_input_table takes : ", end_time - start_time)

    return input_and_equiv_encoding


def get_label_table():
    start_time = datetime.datetime.now()

    query = "select * from diacritics_and_undiacritized_letter_one_hot_encoding"
    cur.execute(query)

    labels_and_equiv_encoding = cur.fetchall()
    labels_and_equiv_encoding = np.array(labels_and_equiv_encoding)

    end_time = datetime.datetime.now()
    print("get_db_label_table takes : ", end_time - start_time)

    return labels_and_equiv_encoding


def load_nn_input_dataset(data_table):
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

    print("create_nn_input_data takes : ", end_time - start_time)

    return nn_input_np_array


def load_nn_labels_dataset(data_table):
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

    print("create_nn_labels takes : ", end_time - start_time)

    return nn_label_np_array


def load_nn_seq_lengths(data_table):

    start_time = datetime.datetime.now()

    nn_seq_lengths = np.unique(data_table, return_counts=True)

    end_time = datetime.datetime.now()

    print("load_nn_seq_lengths takes : ", end_time - start_time)

    return nn_seq_lengths


if __name__ == "__main__":

    establish_db_connection()
    dataset = load_dataset_table('training')
    load_nn_input_dataset(dataset[:, [0, 8]])
    load_nn_labels_dataset(dataset[:, [0, 1]])
    load_nn_seq_lengths(dataset[:, [3]])
