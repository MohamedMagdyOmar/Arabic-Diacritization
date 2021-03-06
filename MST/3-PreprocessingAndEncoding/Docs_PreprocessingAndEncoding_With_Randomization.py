# -*- coding: utf-8 -*-
import re
import unicodedata
import MySQLdb
import math
import os
import random
import copy

# 1

listOfPunctuationSymbols = [u' .', u'.', u' :', u'«', u'»', u'،', u'؛', u'؟', u'.(', u').', u':(', u'):', u'» .', u'».']
listOfArabicDiacriticsUnicode = [["064b", "064c", "064d", "064e", "064f", "0650", "0651", "0652", "0670"],
                                 [1, 2, 3, 4, 5, 6, 8, 7, 9]]
sentenceCount = 0


class DbObject:
    diacritizedCharacter = "",
    diacritizedWord = ""

    undiacritizedCharacter = "",
    undiacritizedWord = ""
    encoded_input = ""
    encoded_input_in_hex_format = ""
    encoded_output = ""
    encoded_output_in_hex_format = ""

    diacritics = "",
    location = "",
    sentenceNumber = 0

    def __init__(self):
        self.diacritizedCharacter = ""
        self.diacritizedWord = ""

        self.undiacritizedCharacter = ""
        self.undiacritizedWord = ""

        self.diacritics = ""
        self.sentenceNumber = ""

        self.encoded_input = ""
        self.encoded_input_in_hex_format = ""

        self.encoded_output = ""
        self.encoded_output_in_hex_format = ""

        self.location = ""


def get_all_files():
    list_of_files_paths = []
    list_of_doc_name = []
    for file_name in os.listdir("D:\MasterRepo\MSTRepo\PaperCorpus\Doc"):
        if file_name.endswith(".txt"):
            list_of_files_paths.append(os.path.join("D:\MasterRepo\MSTRepo\PaperCorpus\Doc", file_name))
            list_of_doc_name.append(file_name)
            print(os.path.join("D:\MasterRepo\MSTRepo\PaperCorpus\Doc", file_name))

    return list_of_files_paths, list_of_doc_name


def read_doc(each_doc, list_of_paths, list_of_docs):

    f = open(list_of_paths[each_doc], 'r')
    document_name = list_of_docs[each_doc]
    data = f.readlines()
    f.close()

    return document_name, data


def extract_and_clean_words_from_doc(data):
    list_of_words = []
    for eachSentence in data:
        words_in_sentence = eachSentence.split()
        for word in words_in_sentence:
            word = word.decode('utf-8', 'ignore') # variable line
            word = re.sub(u'[-;}()/]', '', word)
            word = re.sub(u'[-;}()0123456789/]', '', word)
            word = re.sub(u'["{"]', '', word)
            word = re.sub(u'[:]', ' :', word)

            word = re.sub(u'[ـ]', '', word)
            word = re.sub(u'[`]', '', word)
            word = re.sub(u'[[]', '', word)
            word = re.sub(u'[]]', '', word)
            word = re.sub(u'[L]', '', word)
            word = re.sub(u'[+]', '', word)
            word = re.sub(u'[!]', '', word)
            word = re.sub(u'[\']', '', word)
            word = re.sub(u'[...]', '', word)
            word = re.sub(u'[*]', '', word)
            word = re.sub(u'[&]', '', word)
            word = re.sub(u'[_]', '', word)
            word = re.sub(u'[q]', '', word)
            word = re.sub(u'[u]', '', word)
            word = re.sub(u'[o]', '', word)
            word = re.sub(u'[t]', '', word)
            if not (word == u''):
                list_of_words.append(word)

    return list_of_words


def bind_words_with_sentence_number_in_this_doc(doc_name, list_of_words, raw_data):
    global wordCount
    global sentenceCount
    list_of_words_in_doc_and_corresponding_sentence_number = []

    if doc_name != 'quran-simple.txt' and not (doc_name.startswith('ANN2002')):
        for word in list_of_words:
            if not (word in listOfPunctuationSymbols):
                if word.find(u'.') != -1:
                    wordCount += 1
                    word = re.sub(u'[.]', '', word)
                    if word != u'':
                        list_of_words_in_doc_and_corresponding_sentence_number.append([word, sentenceCount])

                    list_of_words_in_doc_and_corresponding_sentence_number.append(["eos", sentenceCount])
                    sentenceCount += 1
                else:
                    wordCount += 1
                    list_of_words_in_doc_and_corresponding_sentence_number.append([word, sentenceCount])
                    list_of_words_in_doc_and_corresponding_sentence_number.append(["space", sentenceCount])
            else:
                sentenceCount += 1
                list_of_words_in_doc_and_corresponding_sentence_number.append(["bos", sentenceCount])

    else:
        sentenceCount = 0
        wordCount = 0
        for eachVersus in raw_data:
            sentenceCount += 1
            each_word_in_versus = eachVersus.split()
            list_of_words_in_doc_and_corresponding_sentence_number.append(["bos", sentenceCount])
            for eachWord in each_word_in_versus:
                eachWord = eachWord.decode('utf-8', 'ignore')
                wordCount += 1
                list_of_words_in_doc_and_corresponding_sentence_number.append([eachWord, sentenceCount])
                list_of_words_in_doc_and_corresponding_sentence_number.append(["space", sentenceCount])

            list_of_words_in_doc_and_corresponding_sentence_number.pop()
            list_of_words_in_doc_and_corresponding_sentence_number.append(["eos", sentenceCount])

    return list_of_words_in_doc_and_corresponding_sentence_number


def get_list_of_undiacritized_word_from_diacritized_word(list_of_extracted_words_without_numbers):

    listOfUnDiacritizedWord = []

    for word in list_of_extracted_words_without_numbers:

        if word[0] != "space" and word[0] != "bos" and word[0] != "eos":
            if not word[0] in listOfPunctuationSymbols:

                if word[0].find(u'.') != -1:
                    word[0] = re.sub(u'[.]', '', word[0])

                #word[0] = word[0].decode('utf-8', 'ignore')
                nfkd_form = unicodedata.normalize('NFKD', word[0])

                word[0] = u"".join([c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٔ' or c == u'ٕ'])
                listOfUnDiacritizedWord.append(word)
        else:
            listOfUnDiacritizedWord.append(word)

    return listOfUnDiacritizedWord


def character_encoder(list_of_extracted_words_without_numbers):
    listOfEncodedCharacters = []
    listOfEncodedCharactersInHexFormat = []

    for word in list_of_extracted_words_without_numbers:

        if word[0] != "space" and word[0] != "eos" and word[0] != "bos":
            if not word[0] in listOfPunctuationSymbols:
                if word[0].find(u'.') != -1:
                    word[0] = re.sub(u'[.]', '', word[0])

                letterFoundFlag = False
                prevCharWasDiac = False

                for c in word[0]:
                    if not unicodedata.combining(c):  # letter
                        letterFoundFlag = True
                        hexAsString = hex(ord(c))[2:].zfill(4)
                        integer = int(hexAsString, 16)
                        maskedInt = integer & 255
                        shiftedInt = maskedInt << 4
                        listOfEncodedCharactersInHexFormat.append(hex(shiftedInt))
                        listOfEncodedCharacters.append(bin(shiftedInt)[2:].zfill(16))

                    elif letterFoundFlag and c != u'ٔ' and c != u'ٕ':  # first diacritization
                        prevCharWasDiac = True
                        letterFoundFlag = False

                        hexDiacAsString = hex(ord(c))[2:].zfill(4)

                        integerDiac = listOfArabicDiacriticsUnicode[1][
                            listOfArabicDiacriticsUnicode[0].index(hexDiacAsString)]
                        integerDiacAfterORing = shiftedInt | integerDiac
                        listOfEncodedCharacters.pop()
                        listOfEncodedCharacters.append(bin(integerDiacAfterORing)[2:].zfill(16))

                        listOfEncodedCharactersInHexFormat.pop()
                        listOfEncodedCharactersInHexFormat.append(hex(integerDiacAfterORing))

                    elif prevCharWasDiac and c != u'ٔ' and c != u'ٕ':  # second diacritization

                        letterFoundFlag = False
                        prevCharWasDiac = False

                        hexSecDiacAsString = hex(ord(c))[2:].zfill(4)

                        integerSecDiac = listOfArabicDiacriticsUnicode[1][
                            listOfArabicDiacriticsUnicode[0].index(hexSecDiacAsString)]
                        integerSecDiacAfterORing = integerDiacAfterORing | integerSecDiac
                        listOfEncodedCharacters.pop()
                        listOfEncodedCharacters.append(bin(integerSecDiacAfterORing)[2:].zfill(16))
                        listOfEncodedCharactersInHexFormat.pop()
                        listOfEncodedCharactersInHexFormat.append(hex(integerSecDiacAfterORing))
        else:
            listOfEncodedCharacters.append(word[0])
            listOfEncodedCharactersInHexFormat.append(word[0])

    return listOfEncodedCharacters, listOfEncodedCharactersInHexFormat


def extract_each_character_from_word_with_its_diacritization(list_of_extracted_words_and_corresponding_sentence_number, un_diacritized_words):

    DbList = []
    letterFoundFlag = False
    prevCharWasDiac = False

    loopCount = 0
    overall = ""
    diacritics_only_overall = ""

    for eachObject, un_diacritized_word in zip(list_of_extracted_words_and_corresponding_sentence_number, un_diacritized_words):
        diacritizedWord = eachObject[0]
        sentenceNumber = eachObject[1]
        first_char_flag = True

        loopCount += 1
        if eachObject[0] != 'space' and eachObject[0] != 'bos' and eachObject[0] != 'eos':

            spaChar = unicodedata.normalize('NFC', eachObject[0])

            for c in spaChar:

                if not unicodedata.combining(c):
                    letterFoundFlag = True
                    overall = c
                    comp = unicodedata.normalize('NFC', c)
                    newObject = DbObject()

                    newObject.diacritizedCharacter = comp
                    newObject.diacritizedWord = diacritizedWord

                    newObject.undiacritizedCharacter = c
                    newObject.undiacritizedWord = un_diacritized_word[0]

                    newObject.diacritics = ""
                    newObject.sentenceNumber = sentenceNumber

                    if first_char_flag:
                        newObject.location = "first"
                        first_char_flag = False
                    else:
                        newObject.location = "middle"

                    DbList.append(newObject)

                elif letterFoundFlag and c != u'ٔ' and c != u'ٕ':

                    prevCharWasDiac = True
                    letterFoundFlag = False
                    overall += c
                    diacritics_only_overall = c

                    newObject.diacritizedCharacter = unicodedata.normalize('NFC', overall)
                    newObject.diacritics = unicodedata.normalize('NFC', diacritics_only_overall)
                    newObject.sentenceNumber = sentenceNumber

                    DbList.pop()
                    DbList.append(newObject)

                elif prevCharWasDiac and c != u'ٔ' and c != u'ٕ':  # second diacritization

                    letterFoundFlag = False
                    prevCharWasDiac = False
                    overall += c
                    diacritics_only_overall += c

                    newObject.diacritizedCharacter = unicodedata.normalize('NFC', overall)
                    newObject.diacritics = unicodedata.normalize('NFC', diacritics_only_overall)
                    newObject.sentenceNumber = sentenceNumber

                    DbList.pop()
                    DbList.append(newObject)
        else:
            newObject = DbObject()

            newObject.diacritizedCharacter = eachObject[0]
            newObject.diacritizedWord = eachObject[0]

            newObject.undiacritizedCharacter = eachObject[0]
            newObject.undiacritizedWord = eachObject[0]

            newObject.diacritics = ""
            newObject.sentenceNumber = eachObject[1]
            DbList.append(newObject)

        DbList[-1].location = 'last'


    return DbList


def collect_all_data_in_one_list(db_list, encoded_input, encoded_input_in_hex, encoded_output, encoded_output_in_hex):

    for each_object, each_encoded_input, each_encoded_input_in_hex, each_encoded_output, each_encoded_output_in_hex in zip(db_list, encoded_input, encoded_input_in_hex, encoded_output, encoded_output_in_hex):
        each_object.encoded_input = each_encoded_input
        each_object.encoded_input_in_hex_format = each_encoded_input_in_hex
        each_object.encoded_output = each_encoded_output
        each_object.encoded_output_in_hex_format = each_encoded_output_in_hex

    return db_list


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


def prepare_list_for_randomization(db_data):
    list_of_sentence_objects = []
    intermediate_list = []
    current_sentence_number = 1
    for each_object in db_data:

        if each_object.sentenceNumber == current_sentence_number:
            intermediate_list.append(each_object)
        else:
            list_of_sentence_objects.append(intermediate_list)
            intermediate_list = []
            intermediate_list.append(each_object)
            current_sentence_number += 1
    list_of_sentence_objects.append(intermediate_list)

    return list_of_sentence_objects


def randomize_data(data):

    randomized_characters = []
    randomized_sentence = random.sample(data, len(data))
    for each_sentence_object in randomized_sentence:
        for each_letter_object in each_sentence_object:
            randomized_characters.append(each_letter_object)

    return randomized_characters, randomized_sentence


def push_data_into_db(doc, data_chars, db_sentences, list_of_words_and_corresponding_sentence_number):

    required_percentage_for_validation = math.ceil((len(db_sentences) * 12) / 100)
    training_counter = len(db_sentences) - required_percentage_for_validation

    # Part A : filling "Encoded Words" Table
    for x in range(0, len(data_chars)):

        cur.execute(
            "INSERT INTO EncodedWords("
            "InputSequenceEncodedWords,"
            "TargetSequenceEncodedWords,"
            "diacritizedCharacter,"
            "undiacritizedCharacter,"
            "InputSequenceEncodedWordsInHexFormat,"
            "TargetSequenceEncodedWordsInHexFormat, "
            "Diacritics) "
            "VALUES ( "
            "%s,%s,%s,%s,%s,%s,%s)",
            (data_chars[x].encoded_input,
             data_chars[x].encoded_output,
             data_chars[x].diacritizedCharacter,
             data_chars[x].undiacritizedCharacter,
             data_chars[x].encoded_input_in_hex_format,
             data_chars[x].encoded_output_in_hex_format,
             data_chars[x].diacritics))

    for each_sent in db_sentences:
        if training_counter >= 0:
            for each_letter_object in each_sent:
                cur.execute(
                    "INSERT INTO ParsedDocument("
                    "DocName, "
                    "UnDiacritizedCharacter,"
                    "DiacritizedCharacter,"
                    "LetterType,"
                    "SentenceNumber, "
                    "Word, "
                    "InputSequenceEncodedWords,"
                    "TargetSequenceEncodedWords, "
                    "InputSequenceEncodedWordsInHexFormat,"
                    "TargetSequenceEncodedWordsInHexFormat, "
                    "Diacritics, "
                    "UnDiacritizedWord, "
                    "location) VALUES (%s,%s,%s,%s,%s,%s,%s,""%s,%s,%s,%s,%s,%s)",
                    (doc,
                     each_letter_object.undiacritizedCharacter,
                     each_letter_object.diacritizedCharacter,
                     'training',
                     each_letter_object.sentenceNumber,
                     each_letter_object.diacritizedWord,
                     each_letter_object.encoded_input,
                     each_letter_object.encoded_output,
                     each_letter_object.encoded_input_in_hex_format,
                     each_letter_object.encoded_output_in_hex_format,
                     each_letter_object.diacritics,
                     each_letter_object.undiacritizedWord,
                     each_letter_object.location))
            training_counter -= 1
        else:
            for each_letter_object in each_sent:
                cur.execute(
                    "INSERT INTO ParsedDocument("
                    "DocName, "
                    "UnDiacritizedCharacter,"
                    "DiacritizedCharacter,"
                    "LetterType,"
                    "SentenceNumber, "
                    "Word, "
                    "InputSequenceEncodedWords,"
                    "TargetSequenceEncodedWords, "
                    "InputSequenceEncodedWordsInHexFormat,"
                    "TargetSequenceEncodedWordsInHexFormat, "
                    "Diacritics, "
                    "UnDiacritizedWord, "
                    "location) VALUES (%s,%s,%s,%s,%s,%s,%s,""%s,%s,%s,%s,%s,%s)",
                    (doc,
                     each_letter_object.undiacritizedCharacter,
                     each_letter_object.diacritizedCharacter,
                     'testing',
                     each_letter_object.sentenceNumber,
                     each_letter_object.diacritizedWord,
                     each_letter_object.encoded_input,
                     each_letter_object.encoded_output,
                     each_letter_object.encoded_input_in_hex_format,
                     each_letter_object.encoded_output_in_hex_format,
                     each_letter_object.diacritics,
                     each_letter_object.undiacritizedWord,
                     each_letter_object.location))

    for each_word in list_of_words_and_corresponding_sentence_number:
        cur.execute(
            "INSERT INTO ListOfWordsAndSentencesInEachDoc(word,SentenceNumber,DocName) VALUES (%s,%s,%s)",
            (each_word[0], each_word[1], doc))

    db.commit()
    db.close()


if __name__ == "__main__":
    listOfFilesPaths, ListOfDocs = get_all_files()

    for eachDoc in range(0, len(listOfFilesPaths)):
        doc_name = listOfFilesPaths[eachDoc]
        selected_doc, raw_data = read_doc(eachDoc, listOfFilesPaths, ListOfDocs)
        cleaned_data = extract_and_clean_words_from_doc(raw_data)
        listOfWordsAndCorrespondingSentenceNumber = bind_words_with_sentence_number_in_this_doc(selected_doc, cleaned_data, raw_data)
        listOfUndiacritizedWords = get_list_of_undiacritized_word_from_diacritized_word(copy.deepcopy(listOfWordsAndCorrespondingSentenceNumber))

        encodedInput, encodedInputInHexFormat = character_encoder(listOfUndiacritizedWords)
        encodedTarget, encodedTargetInHexFormat = character_encoder(listOfWordsAndCorrespondingSentenceNumber)
        dbList = extract_each_character_from_word_with_its_diacritization(
            listOfWordsAndCorrespondingSentenceNumber, listOfUndiacritizedWords)

        data = collect_all_data_in_one_list(dbList, encodedInput, encodedInputInHexFormat, encodedTarget, encodedTargetInHexFormat)

        connect_to_db()

        list_to_be_randomized = prepare_list_for_randomization(data)
        list_of_randomized_characters,  list_of_randomized_sentences = randomize_data(list_to_be_randomized)
        push_data_into_db(selected_doc, list_of_randomized_characters, list_of_randomized_sentences, listOfWordsAndCorrespondingSentenceNumber)


