# -*- coding: utf-8 -*-

# purpose of below code is :
#  1- is to clean the document from punctuation symbol
#  2- get document statistics
#  3- create comma separated diacritized and undiacritized documents


import unicodedata
import re
import MySQLdb

docName = "الجامع الصحيح المسمى صحيح مسلم";
listOfUndiacritizedCharacter = [];
listOfDiacritizedCharacter = [];

listOfPunctuationBySymbol = [' .', ' :', '«', '»', '،', '؛', '؟', '.(', ').', ':(', '):', '» .','».']
listOfPunctuationByCodePoint = ['060C', '061B', '061F', '003A', '002E']

f = open("/home/mohamed/Desktop/الجامع الصحيح المسمى صحيح مسلم.txt", 'r')
read_data = f.readlines()
f.close()

listOfWords = []
listOfWordsInSent = []
ListOfWordsWithPunctuation = []
wordCount = 0

# removing punctuation and extracting words from Doc
for eachSentence in read_data:
    wordsInSentence = eachSentence.split()
    for word in wordsInSentence:
        word = re.sub('[-;}()0123456789/]', '', word)
        word = re.sub('["{"]', '', word)
        word = re.sub('[:]', ' :', word)

        if not (word == ''):
            listOfWords.append(word)

sentenceCount = 1

# Creating List of words in sent with corresponding sentence count.
for word in listOfWords:

    if not (word in listOfPunctuationBySymbol):
        if(word.find('.')!=-1):
            ListOfWordsWithPunctuation.append([word, sentenceCount])

            word = re.sub('[.]', '', word)
            wordCount += 1
            listOfWordsInSent.append([word, sentenceCount])
            sentenceCount += 1
        else:
            wordCount += 1
            listOfWordsInSent.append([word, sentenceCount])
            ListOfWordsWithPunctuation.append([word, sentenceCount])
    else:
        ListOfWordsWithPunctuation.append([word, sentenceCount])
        sentenceCount += 1

listOfUnDiacritizedWord = []
letterCount = 0
prevCharWasDiac = False
letterFoundFlag = False
lettersWithNoDiac = 0
lettersWithOneDiac = 0
lettersWithTwoDiac = 0

listOfLettersWith2Diac = []

# removing diacritization from words, and count number of letters with and without diacritization.
for word in listOfWords:
    if not word in listOfPunctuationBySymbol:

        word = word.decode('utf-8', 'ignore')
        nfkd_form = unicodedata.normalize('NFKD', word)

        unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
        listOfUnDiacritizedWord.append(unDiacritizedWord)

        letterFoundFlag = False
        prevCharWasDiac = False

        for c in nfkd_form:

            if not unicodedata.combining(c):
                letterFoundFlag = True
                lettersWithNoDiac += 1
                listOfUndiacritizedCharacter.append(c);

            elif letterFoundFlag and c != u'ٔ' and c != u'ٕ':
                lettersWithOneDiac += 1
                prevCharWasDiac = True
                lettersWithNoDiac -= 1
                letterFoundFlag = False

            elif prevCharWasDiac and c != u'ٔ' and c != u'ٕ':
                lettersWithTwoDiac += 1
                lettersWithOneDiac -= 1
                letterFoundFlag = False
                prevCharWasDiac = False
                listOfLettersWith2Diac.append(c)

        for eachLetter in unDiacritizedWord:
            letterCount += 1

print 'letter count:', letterCount
print 'word count: ', wordCount
print 'sentence count:', sentenceCount

print 'number of letters with no diac', float(lettersWithNoDiac)
print 'number of letters with One diac', float(lettersWithOneDiac)
print 'number of letters with Two diac', float(lettersWithTwoDiac)

print 'letters per word:', (float(letterCount) / float(wordCount))
print 'words per sentence:', float(wordCount) / float(sentenceCount)
print 'letters without diac:', (float(lettersWithNoDiac) / float(letterCount)) * 100
print 'letters with One diac:', (float(lettersWithOneDiac) / float(letterCount)) * 100
print 'letters with Two diac:', (float(lettersWithTwoDiac) / float(letterCount)) * 100

finalWordList = []
for word in listOfWordsInSent:
    word = word[0].decode('utf-8', 'ignore')
    finalWordList.append(word)

f = open("/home/mohamed/Desktop/Jawhara_Elnayera_SentenceParsed_With_Punct.txt", 'w')

rowsCount = 0
columnsCount = len(ListOfWordsWithPunctuation[0])
CurrentSentenceNumber = ListOfWordsWithPunctuation[0][1]
currentRow = 1

while rowsCount < len(ListOfWordsWithPunctuation):
    if ListOfWordsWithPunctuation[rowsCount][1] == currentRow:
        f.write(ListOfWordsWithPunctuation[rowsCount][0])
        f.write(" ")
        rowsCount += 1
    else:
        currentRow += 1
        f.write('\n')
        f.write('\n')

f.close()

f = open("/home/mohamed/Desktop/Jawhara_Elnayera_SentenceParsed_Without_Punct.txt", 'w')

rowsCount = 0
columnsCount = len(ListOfWordsWithPunctuation[0])
CurrentSentenceNumber = ListOfWordsWithPunctuation[0][1]
currentRow = 1

while rowsCount < len(listOfWordsInSent):
    if listOfWordsInSent[rowsCount][1] == currentRow:
        f.write(listOfWordsInSent[rowsCount][0])
        f.write(" ")
        rowsCount += 1
    else:
        currentRow += 1
        f.write('\n')
        f.write('\n')

f.close()

