# -*- coding: utf-8 -*-
import unicodedata
import MySQLdb
import MySQLdb.cursors
import data_helper as dp
import DBHelperMethod
import numpy as np
import copy
import datetime
doc = 'quran-simple.txt'


def connect_to_db():
    global db
    db = MySQLdb.connect(host="127.0.0.1",  # your host, usually localhost
                         user="root",  # your username
                         passwd="Islammega88",  # your password
                         db="MSTDB",  # name of the data base
                         use_unicode=True,
                         charset="utf8",
                         init_command='SET NAMES UTF8')

    global cur
    cur = db.cursor()


def load_training_data():

    parsed_document = DBHelperMethod.load_data_set()
    sent_list = DBHelperMethod.get_all_sentences()
    letter_type_sentence_number = parsed_document[:, 2:4]

    unique_rows = np.unique(letter_type_sentence_number, axis=0)
    connect_to_db()
    for each_row in sent_list:
        rows, cols = np.where(unique_rows == str(each_row[2]))

        cur.execute(
            "INSERT INTO test(word,SentenceNumber,DocName, wordtype) VALUES (%s,%s,%s,%s)",
            (each_row[1], each_row[2], doc, unique_rows[rows, 0][0]))

    db.commit()
    db.close()


if __name__ == "__main__":

    load_training_data()
