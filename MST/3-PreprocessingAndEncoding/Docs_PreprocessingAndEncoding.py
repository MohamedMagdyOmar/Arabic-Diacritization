# -*- coding: utf-8 -*-
import re
import unicodedata
import MySQLdb
import math
import os
import random
import datetime
# 1
diacritizedCharacter = []
DiacriticsOnly = []
unDiacritizedCharacter = []
listOfDBWords = []
listOfDbSentenceNumber = []
listOfPunctuationBySymbol = [u' .', u'.', u' :', u'«', u'»', u'،', u'؛', u'؟', u'.(', u').', u':(', u'):', u'» .', u'».']
final_ListOfUndiacritized_Word = []
listOfArabicDiacriticsUnicode = [["064b", "064c", "064d", "064e", "064f", "0650", "0651", "0652", "0670"],
                                 [1, 2, 3, 4, 5, 6, 8, 7, 9]]

docname = ""
def declareGlobalVariables():
    global wordCount
    wordCount = 0
    global sentenceCount
    sentenceCount = 1

    global list_of_all_sentence
    list_of_all_sentence = []


def findFiles():
    global listOfFilesPathes
    global listOfDocName
    listOfFilesPathes = []
    listOfDocName = []
    for file in os.listdir("D:\MasterRepo\MSTRepo\PaperCorpus\Doc"):
        if file.endswith(".txt"):
            listOfFilesPathes.append(os.path.join("D:\MasterRepo\MSTRepo\PaperCorpus\Doc", file))
            listOfDocName.append(file)
            print(os.path.join("D:\MasterRepo\MSTRepo\PaperCorpus\Doc", file))


def readDoc(eachdoc):
    global read_data
    global docName
    f = open(listOfFilesPathes[eachdoc], 'r')
    docName = listOfDocName[eachdoc]
    read_data = f.readlines()
    f.close()


def extractAndCleanWordsFromDoc():
    global listOfWords
    listOfWords = []
    for eachSentence in read_data:
        wordsInSentence = eachSentence.split()
        for word in wordsInSentence:
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
            word = re.sub(u'ٰ', '', word)
            word = re.sub(u'ـ', '', word)
            if not (word == u''):
                listOfWords.append(word)


def extractSentencesFromDoc():
    global wordCount
    global listOfWordsInSent
    global ListOfWordsWithPunctuation
    global sentenceCount
    #wordCount = 0
    listOfWordsInSent = []
    ListOfWordsWithPunctuation = []
    #sentenceCount = 1

    if docName != 'quran-simple.txt' and not (docName.startswith('ANN2002')):
        for word in listOfWords:

            if not (word in listOfPunctuationBySymbol):
                if word.find(u'.') != -1:
                    wordCount += 1

                    ListOfWordsWithPunctuation.append([word, sentenceCount])

                    word = re.sub(u'[.]', '', word)
                    if word != u'':
                        listOfWordsInSent.append([word, sentenceCount])
                    sentenceCount += 1
                else:
                    wordCount += 1
                    listOfWordsInSent.append([word, sentenceCount])
                    ListOfWordsWithPunctuation.append([word, sentenceCount])
            else:
                ListOfWordsWithPunctuation.append([word, sentenceCount])
                sentenceCount += 1
    else:
        sentenceCount = 0
        wordCount = 0
        for eachVersus in read_data:
            sentenceCount += 1
            eachWordInVersus = eachVersus.split()
            for eachWord in eachWordInVersus:
                wordCount += 1
                listOfWordsInSent.append([eachWord, sentenceCount])


def encodingDiacritizedCharacter():
    global listOfTargetSequenceEncodedWords
    global listOfInputSequenceEncodedWordsInHexFormat
    global listOfTargetSequenceEncodedWordsInHexFormat
    global listOfInputSequenceEncodedWords
    global listOfUnDiacritizedWord
    listOfInputSequenceEncodedWords = []
    listOfUnDiacritizedWord = []
    listOfTargetSequenceEncodedWords = []
    listOfInputSequenceEncodedWordsInHexFormat = []
    listOfTargetSequenceEncodedWordsInHexFormat = []

    for word in listOfWords:
        if not word in listOfPunctuationBySymbol:
            if word.find(u'.') != -1:
                word = re.sub(u'[.]', '', word)
#            word = word.decode('utf-8', 'ignore') may be need to return itback

            nfkd_form = unicodedata.normalize('NFKD', word)

            unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
            listOfUnDiacritizedWord.append(unDiacritizedWord)

            letterFoundFlag = False
            prevCharWasDiac = False

            for c in word:

                if not unicodedata.combining(c):  # letter
                    letterFoundFlag = True

                    hexAsString = hex(ord(c))[2:].zfill(4)

                    binaryAsString = bin(int(hexAsString, 16))[2:].zfill(16)
                    integer = int(hexAsString, 16)
                    maskedInt = integer & 255
                    maskedBinaryAsString = bin(integer & 255)[2:].zfill(16)
                    shiftedInt = maskedInt << 4
                    shiftedIntInBin = bin(shiftedInt)

                    listOfTargetSequenceEncodedWordsInHexFormat.append(hex(shiftedInt))
                    listOfTargetSequenceEncodedWords.append(bin(shiftedInt)[2:].zfill(16))
                    listOfInputSequenceEncodedWordsInHexFormat.append(hex(shiftedInt))
                    listOfInputSequenceEncodedWords.append(str(bin(shiftedInt)[2:].zfill(16)))

                elif letterFoundFlag and c != u'ٔ' and c != u'ٕ':  # first diacritization
                    prevCharWasDiac = True
                    letterFoundFlag = False

                    hexDiacAsString = hex(ord(c))[2:].zfill(4)
                    if hexDiacAsString == '0670' :
                        x = 1
                    binaryAsString = bin(int(hexDiacAsString, 16))[2:].zfill(16)
                    integerDiac = listOfArabicDiacriticsUnicode[1][
                        listOfArabicDiacriticsUnicode[0].index(hexDiacAsString)]
                    integerDiacAfterORing = shiftedInt | integerDiac
                    listOfTargetSequenceEncodedWords.pop()
                    listOfTargetSequenceEncodedWords.append(bin(integerDiacAfterORing)[2:].zfill(16))

                    listOfTargetSequenceEncodedWordsInHexFormat.pop()
                    listOfTargetSequenceEncodedWordsInHexFormat.append(hex(integerDiacAfterORing))
                elif prevCharWasDiac and c != u'ٔ' and c != u'ٕ':  # second diacritization

                    letterFoundFlag = False
                    prevCharWasDiac = False

                    hexSecDiacAsString = hex(ord(c))[2:].zfill(4)

                    integerSecDiac = listOfArabicDiacriticsUnicode[1][
                        listOfArabicDiacriticsUnicode[0].index(hexSecDiacAsString)]
                    integerSecDiacAfterORing = integerDiacAfterORing | integerSecDiac
                    listOfTargetSequenceEncodedWords.pop()
                    listOfTargetSequenceEncodedWords.append(bin(integerSecDiacAfterORing)[2:].zfill(16))
                    listOfTargetSequenceEncodedWordsInHexFormat.pop()
                    listOfTargetSequenceEncodedWordsInHexFormat.append(hex(integerSecDiacAfterORing))


def encodingunDiacritizedCharacter():
    global listOfUnDiacritizedWord
    global listOfInputSequenceEncodedWords
    listOfUnDiacritizedWord = []
    listOfInputSequenceEncodedWords = []

    for word in listOfWords:
        if not word in listOfPunctuationBySymbol:

            if word.find('.') != -1:
                word = re.sub('[.]', '', word)

            word = word.decode('utf-8', 'ignore')
            nfkd_form = unicodedata.normalize('NFKD', word)

            unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
            listOfUnDiacritizedWord.append(unDiacritizedWord)

            for c in word:

                if not unicodedata.combining(c):  # letter
                    letterFoundFlag = True

                    hexAsString = hex(ord(c))[2:].zfill(4)

                    binaryAsString = bin(int(hexAsString, 16))[2:].zfill(16)
                    integer = int(hexAsString, 16)
                    maskedInt = integer & 255
                    maskedBinaryAsString = bin(integer & 255)[2:].zfill(16)
                    shiftedInt = maskedInt << 4
                    shiftedIntInBin = bin(shiftedInt)
                    listOfInputSequenceEncodedWords.append(bin(shiftedInt)[2:].zfill(16))


def convertToString():
    for item in range(0, len(listOfInputSequenceEncodedWords)):
        listOfInputSequenceEncodedWords[item] = str(listOfInputSequenceEncodedWords[item])


first = ""
second = ""
third = ""
overall = ""
#  = unicodedata.normalize('NFC', word)

def extractEachCharacterFromWordWithItsDiacritization():
    letterFoundFlag = False
    prevCharWasDiac = False
    loopCount = 0
    first = ""
    second = ""
    third = ""
    overall = ""
    diacritics_only_overall = ""

    for word in listOfWords:

        if not word in listOfPunctuationBySymbol:

            if word.find(u'.') != -1:
                word = re.sub(u'[.]', '', word)

#            word = word.decode('utf-8', 'ignore') may be return back

            # removing diacritization from characters
            nfkd_form = unicodedata.normalize('NFKD', word)
            unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
            try:
                sentenceNumber = listOfWordsInSent[loopCount][1]
            except:
                x = 1
            loopCount += 1

            spaChar = unicodedata.normalize('NFC', word)
            for c in spaChar:

                if not unicodedata.combining(c):
                    first = c
                    letterFoundFlag = True
                    overall = c
                    comp = unicodedata.normalize('NFC', c)
                    diacritizedCharacter.append(comp)

                    listOfDbSentenceNumber.append(sentenceNumber)

                    listOfDBWords.append(word)

                    listOfUnDiacritizedWord.append(unDiacritizedWord)
                    unDiacritizedCharacter.append(c)
                    DiacriticsOnly.append("")
                    unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c) or c == u'ٔ' or c == u'ٕ'])
                    final_ListOfUndiacritized_Word.append(unDiacritizedWord)

                elif letterFoundFlag and c != u'ٔ' and c != u'ٕ':
                    second = c
                    prevCharWasDiac = True
                    letterFoundFlag = False
                    overall += c
                    diacritics_only_overall = c

                    comp = unicodedata.normalize('NFC', overall)
                    comp_diacritics_Only = unicodedata.normalize('NFC', diacritics_only_overall)

                    diacritizedCharacter.pop()
                    diacritizedCharacter.append(comp)

                    DiacriticsOnly.pop()
                    DiacriticsOnly.append(comp_diacritics_Only)
                elif prevCharWasDiac and c != u'ٔ' and c != u'ٕ':  # second diacritization
                    third = c
                    letterFoundFlag = False
                    prevCharWasDiac = False
                    overall += c
                    diacritics_only_overall += c

                    comp = unicodedata.normalize('NFC', overall)
                    comp_diacritics_Only = unicodedata.normalize('NFC', diacritics_only_overall)

                    diacritizedCharacter.pop()
                    diacritizedCharacter.append(comp)

                    DiacriticsOnly.pop()
                    DiacriticsOnly.append(comp_diacritics_Only)
                    # for word in listOfUnDiacritizedWord:
                    # for char in word:
                    #  unDiacritizedCharacter.append(char)


def connectToDB():
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


def pushDataIntoDB():
    requiredPercentageForValidation = math.ceil((len(listOfInputSequenceEncodedWords) * 15) / 100)
    trainingCounter = len(listOfInputSequenceEncodedWords) - (requiredPercentageForValidation * 2)
    isTrainingDataIsFinished = False
    isValidationDataIsFinished = False
    # Part A : filling "Encoded Words" Table
    for x in range(0, len(listOfInputSequenceEncodedWords)):
        cur.execute(
            "INSERT INTO EncodedWords(InputSequenceEncodedWords,TargetSequenceEncodedWords,diacritizedCharacter,"
            "undiacritizedCharacter,InputSequenceEncodedWordsInHexFormat,TargetSequenceEncodedWordsInHexFormat, "
            "Diacritics) "
            "VALUES ( "
            "%s,%s,%s,%s,%s,%s,%s)",
            (listOfInputSequenceEncodedWords[x], listOfTargetSequenceEncodedWords[x], diacritizedCharacter[x],
             unDiacritizedCharacter[x], listOfInputSequenceEncodedWordsInHexFormat[x],
             listOfTargetSequenceEncodedWordsInHexFormat[x], DiacriticsOnly[x]))

        if (trainingCounter >= 0 or prevSentenceNumber == listOfDbSentenceNumber[x]) and \
                (isTrainingDataIsFinished is False):
            prevSentenceNumber = listOfDbSentenceNumber[x]
            trainingCounter -= 1
            cur.execute(
                "INSERT INTO ParsedDocument(DocName, UnDiacritizedCharacter,DiacritizedCharacter,LetterType,"
                "SentenceNumber, "
                "Word, "
                "InputSequenceEncodedWords,TargetSequenceEncodedWords, InputSequenceEncodedWordsInHexFormat,"
                "TargetSequenceEncodedWordsInHexFormat, Diacritics, UnDiacritizedWord) VALUES (%s,%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s,%s,%s)",
                (docName, unDiacritizedCharacter[x], diacritizedCharacter[x], 'testing', listOfDbSentenceNumber[x],
                 listOfDBWords[x], listOfInputSequenceEncodedWords[x], listOfTargetSequenceEncodedWords[x],
                 listOfInputSequenceEncodedWordsInHexFormat[x], listOfTargetSequenceEncodedWordsInHexFormat[x],
                 DiacriticsOnly[x], final_ListOfUndiacritized_Word[x]))
        else:
            isTrainingDataIsFinished = True
            if (requiredPercentageForValidation >= 0 or prevSentenceNumber == listOfDbSentenceNumber[x]) and\
                    (isValidationDataIsFinished is False):
                prevSentenceNumber = listOfDbSentenceNumber[x]
                requiredPercentageForValidation -= 1
                cur.execute(
                    "INSERT INTO ParsedDocument(DocName, UnDiacritizedCharacter,DiacritizedCharacter,LetterType,"
                    "SentenceNumber, "
                    "Word, "
                    "InputSequenceEncodedWords,TargetSequenceEncodedWords, InputSequenceEncodedWordsInHexFormat,"
                    "TargetSequenceEncodedWordsInHexFormat, Diacritics, UnDiacritizedWord) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (docName, unDiacritizedCharacter[x], diacritizedCharacter[x], 'validation', listOfDbSentenceNumber[x],
                    listOfDBWords[x], listOfInputSequenceEncodedWords[x], listOfTargetSequenceEncodedWords[x],
                    listOfInputSequenceEncodedWordsInHexFormat[x], listOfTargetSequenceEncodedWordsInHexFormat[x],
                    DiacriticsOnly[x], final_ListOfUndiacritized_Word[x]))

            else:
                isValidationDataIsFinished = True
                cur.execute(
                    "INSERT INTO ParsedDocument(DocName, UnDiacritizedCharacter,DiacritizedCharacter,LetterType,"
                    "SentenceNumber, "
                    "Word, "
                    "InputSequenceEncodedWords,TargetSequenceEncodedWords, InputSequenceEncodedWordsInHexFormat,"
                    "TargetSequenceEncodedWordsInHexFormat, Diacritics, UnDiacritizedWord) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (docName, unDiacritizedCharacter[x], diacritizedCharacter[x], 'testing',
                     listOfDbSentenceNumber[x],
                     listOfDBWords[x], listOfInputSequenceEncodedWords[x], listOfTargetSequenceEncodedWords[x],
                     listOfInputSequenceEncodedWordsInHexFormat[x], listOfTargetSequenceEncodedWordsInHexFormat[x],
                     DiacriticsOnly[x], final_ListOfUndiacritized_Word[x]))

    for x in range(0, len(listOfWordsInSent)):
        cur.execute(
            "INSERT INTO ListOfWordsAndSentencesInEachDoc(word,SentenceNumber,DocName) VALUES (%s,%s,%s)",
            (listOfWordsInSent[x][0], listOfWordsInSent[x][1], docName))

    db.commit()
    db.close()

def resetAllLists():
    del unDiacritizedCharacter[:]
    del diacritizedCharacter[:]
    del listOfDbSentenceNumber[:]
    del listOfDBWords[:]
    del listOfInputSequenceEncodedWords[:]
    del listOfTargetSequenceEncodedWords[:]
    del listOfInputSequenceEncodedWordsInHexFormat[:]
    del listOfTargetSequenceEncodedWordsInHexFormat[:]
    del listOfWordsInSent[:]


def prepare_list_for_randomization():
    list_of_sentence_content = []
    intermediate_list = []
    #list_of_all_sentence = []
    sentence_number = listOfDbSentenceNumber[0]
    counter = 0

    for current_sentence_number in listOfDbSentenceNumber:
        if current_sentence_number == sentence_number:
            intermediate_list = []
            intermediate_list.append(docName)
            intermediate_list.append(unDiacritizedCharacter[counter])
            intermediate_list.append(diacritizedCharacter[counter])
            intermediate_list.append(listOfDbSentenceNumber[counter])
            intermediate_list.append(listOfDBWords[counter])
            intermediate_list.append(listOfInputSequenceEncodedWords[counter])
            intermediate_list.append(listOfTargetSequenceEncodedWords[counter])
            intermediate_list.append(listOfInputSequenceEncodedWordsInHexFormat[counter])

            intermediate_list.append(listOfTargetSequenceEncodedWordsInHexFormat[counter])
            intermediate_list.append(DiacriticsOnly[counter])
            intermediate_list.append(final_ListOfUndiacritized_Word[counter])
            list_of_sentence_content.append(intermediate_list)
            counter += 1
        else:
            sentence_number += 1
            list_of_all_sentence.append(list_of_sentence_content)
            intermediate_list = []
            list_of_sentence_content = []
            intermediate_list.append(docName)
            intermediate_list.append(unDiacritizedCharacter[counter])
            intermediate_list.append(diacritizedCharacter[counter])
            intermediate_list.append(listOfDbSentenceNumber[counter])
            intermediate_list.append(listOfDBWords[counter])
            intermediate_list.append(listOfInputSequenceEncodedWords[counter])
            intermediate_list.append(listOfTargetSequenceEncodedWords[counter])
            intermediate_list.append(listOfInputSequenceEncodedWordsInHexFormat[counter])

            intermediate_list.append(listOfTargetSequenceEncodedWordsInHexFormat[counter])
            intermediate_list.append(DiacriticsOnly[counter])
            intermediate_list.append(final_ListOfUndiacritized_Word[counter])
            list_of_sentence_content.append(intermediate_list)
            counter += 1
    x = 1

def randomize_Data():
    randomized_Sentence = random.sample(list_of_all_sentence, len(list_of_all_sentence))
    x = 1


if __name__ == "__main__":
    findFiles()
    declareGlobalVariables()
    for eachDoc in range(0, len(listOfFilesPathes)):
        docname = listOfFilesPathes[eachDoc]
        readDoc(eachDoc)
        extractAndCleanWordsFromDoc()
        extractSentencesFromDoc()
        encodingDiacritizedCharacter()
        # encodingunDiacritizedCharacter()
        #  convertToString()
        extractEachCharacterFromWordWithItsDiacritization()
        connectToDB()
        prepare_list_for_randomization()
        randomize_Data()
        pushDataIntoDB()
        resetAllLists()

