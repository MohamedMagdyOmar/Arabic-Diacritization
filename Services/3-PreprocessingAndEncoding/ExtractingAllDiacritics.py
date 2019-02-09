# -*- coding: utf-8 -*-
# 3
import re
import unicodedata
import MySQLdb
import os
import numpy as np

listOfPunctuationBySymbol = [u' .', u'.', u' :', u'«', u'»', u'،', u'؛', u'؟', u'.(', u').', u':(', u'):', u'» .', u'».'
                             ]
DiacriticsOnly = []

listOfArabicDiacriticsUnicode = [["064b", "064c", "064d", "064e", "064f", "0650", "0651", "0652", "0670"],
                                 [1, 2, 3, 4, 5, 6, 8, 7, 9]]


def find_files():
    global listOfFilesPathes
    global listOfDocName
    listOfFilesPathes = []
    listOfDocName = []
    for file in os.listdir("D:\MasterRepo\MSTRepo\PaperCorpus\Doc"):
        if file.endswith(".txt"):
            listOfFilesPathes.append(os.path.join("D:\MasterRepo\MSTRepo\PaperCorpus\Doc", file))
            listOfDocName.append(file)
            print(os.path.join("D:\MasterRepo\MSTRepo\PaperCorpus\Doc", file))


def read_doc(each_doc):
    global read_data
    global docName
    f = open(listOfFilesPathes[each_doc], 'r')
    docName = listOfDocName[each_doc]
    read_data = f.readlines()
    f.close()


def extract_and_clean_words_from_doc():
    global listOfWords
    listOfWords = []
    for eachSentence in read_data:
        wordsInSentence = eachSentence.split()
        for word in wordsInSentence:
            word = word.decode('utf-8', 'ignore')  # variable line

            word = re.sub(u'[-;}()0123456789/]', '', word)
            word = re.sub(u'["{"]', '', word)
            word = re.sub(u'[:]', ' :', word)

            word = re.sub(u'[ـ]', '', word)
            word = re.sub(u'[`]', '', word)
            word = re.sub(u'[[]', '', word)
            word = re.sub(u'[]]', '', word)
            word = re.sub(u'[L]', '', word)
            if not (word == u''):
                listOfWords.append(word)


def extract_diacritization_symbols():
    letterFoundFlag = False
    prevCharWasDiac = False
    overall = ""

    for word in listOfWords:

        if not word in listOfPunctuationBySymbol:

            if word.find(u'.') != -1:
                word = re.sub(u'[.]', '', word)

            spaChar = unicodedata.normalize('NFC', word)
            for c in spaChar:

                if not unicodedata.combining(c):
                    letterFoundFlag = True
                    DiacriticsOnly.append("")
                elif letterFoundFlag and c != u'ٔ' and c != u'ٕ':
                    prevCharWasDiac = True
                    letterFoundFlag = False
                    overall = c
                    comp = unicodedata.normalize('NFC', overall)

                    DiacriticsOnly.pop()
                    DiacriticsOnly.append(comp)
                elif prevCharWasDiac and c != u'ٔ' and c != u'ٕ':  # second diacritization

                    letterFoundFlag = False
                    prevCharWasDiac = False
                    overall += c
                    comp = unicodedata.normalize('NFC', overall)
                    DiacriticsOnly.pop()
                    DiacriticsOnly.append(comp)


def connect_to_db():
    global db
    db = MySQLdb.connect(host="127.0.0.1",  # your host, usually localhost
                         user="root",  # your username
                         passwd="Islammega88",  # your password
                         db="mstdb",  # name of the data base
                         use_unicode=True,
                         charset="utf8",
                         init_command='SET NAMES UTF8')

    global cur
    cur = db.cursor()


def push_data_into_db():
    for x in range(0, len(DiacriticsOnly)):
        cur.execute("INSERT INTO alldiacriticsinalldocuments (Diacritics) VALUES (%s)", [DiacriticsOnly[x]])

    db.commit()
    db.close()


def reset_all_lists():
    del DiacriticsOnly[:]


if __name__ == "__main__":
    find_files()
    for eachDoc in range(0, len(listOfFilesPathes)):
        read_doc(eachDoc)
        extract_and_clean_words_from_doc()
        extract_diacritization_symbols()
        connect_to_db()
        push_data_into_db()
        reset_all_lists()
