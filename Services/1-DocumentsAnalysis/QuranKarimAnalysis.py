# -*- coding: utf-8 -*-
import unicodedata

f = open("/home/mohamed/Desktop/quran-simple.txt", 'r')
read_data = f.readlines()
f.close()

listOfCodePoint = []

for sentence in read_data:
    decodedSentence = sentence.decode('utf-8', 'ignore')
    for eachChar in decodedSentence:
        encodedWord = eachChar.encode('utf-8', 'ignore')
        listOfCodePoint.append(hex(ord(eachChar)))

wordCounter = 0
letterCount = 0
listOfWord = []
listOfDiacritizedWord = []

for eachSentence in read_data:
    wordsInSentence = eachSentence.split()
    for word in wordsInSentence:
        listOfWord.append(word)
    wordCounter += len(eachSentence.split())

listOfLettersWith0Diac = []
listOfLettersWith1Diac = []
listOfLettersWith2Diac = []

prevCharWasDiac = False
letterFoundFlag = False
lettersWithNoDiac = 0
lettersWithOneDiac = 0
lettersWithTwoDiac = 0

for eachSentence in read_data:
    wordsInSentence = eachSentence.split()
    for word in wordsInSentence:
        word = word.decode('utf-8', 'ignore')
        nfkd_form = unicodedata.normalize('NFKD', word)
        unDiacritizedWord = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

        for c in nfkd_form:

            if not unicodedata.combining(c):
                letterFoundFlag = True
                lettersWithNoDiac += 1
                listOfLettersWith0Diac.append(c)

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

        listOfDiacritizedWord.append(unDiacritizedWord)

        for eachLetter in unDiacritizedWord:
            letterCount += 1

print 'letter count:', letterCount
print 'word count: ', wordCounter
print 'verses count:', len(read_data)
print 'letters per word:', (float(letterCount) / float(wordCounter))
print 'words per sentence:', float(wordCounter) / float(len(read_data))
print 'letters without diac:', (float(lettersWithNoDiac) / float(letterCount)) * 100
print 'letters with One diac:', (float(lettersWithOneDiac) / float(letterCount)) * 100
print 'letters with Two diac:', (float(lettersWithTwoDiac) / float(letterCount)) * 100
