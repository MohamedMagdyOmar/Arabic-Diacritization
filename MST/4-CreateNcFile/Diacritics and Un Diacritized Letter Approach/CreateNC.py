# -*- coding: utf-8 -*-
import netCDF4 as netcdf_helpers
import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime

global punchNumber
punchNumber = 0
max_seq_tag_length = 4
import WordLetterProcessingHelperMethod


def create_mysql_connection():
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


def get_all_letters_of_corresponding_dataset_type(type_of_dataset):
    execute_calculate_total_number_Of_sentences_startTime = datetime.datetime.now()

    listOfSelectedLettersAndSentencesQuery = "select UnDiacritizedCharacter, Diacritics, LetterType, " \
                                             "SentenceNumber," \
                                             " Word, InputSequenceEncodedWords, TargetSequenceEncodedWords," \
                                             " DiacritizedCharacter, location " \
                                             "from ParsedDocument where LetterType=" + "'%s'" % type_of_dataset

    cur.execute(listOfSelectedLettersAndSentencesQuery)
    global listOfSelectedLettersAndSentences
    listOfSelectedLettersAndSentences = []
    listOfSelectedLettersAndSentences = cur.fetchall()

    execute_calculate_total_number_of_sentence_end_time = datetime.datetime.now()
    print "get_all_letters_of_corresponding_dataset_type takes : ", execute_calculate_total_number_of_sentence_end_time - \
                                                                    execute_calculate_total_number_Of_sentences_startTime


def execute_unchanged_sql_queries():
    executeChangedSQLQueriesStartTime = datetime.datetime.now()
    listOfUnDiacritizedCharacterQuery = "select * from UnDiacOneHotEncoding"
    cur.execute(listOfUnDiacritizedCharacterQuery)
    global listOfUnDiacritizedCharacter
    listOfUnDiacritizedCharacter = cur.fetchall()

    listOfDiacritizedCharacterQuery = "select * from diacritics_and_undiacritized_letter_one_hot_encoding "
    cur.execute(listOfDiacritizedCharacterQuery)
    global listOfDiacritizedCharacter
    listOfDiacritizedCharacter = cur.fetchall()

    # use this if you are going to predict diacritics only
    # listOfDiacritizedCharacterQuery = "select * from distinctdiacritics "
    # cur.execute(listOfDiacritizedCharacterQuery)
    # global listOfDiacritizedCharacter
    # listOfDiacritizedCharacter = cur.fetchall()

    executeChangedSQLQueriesEndTime = datetime.datetime.now()
    print "executeChangedSQLQueries takes : ", executeChangedSQLQueriesEndTime - executeChangedSQLQueriesStartTime


def create_netcdf_label():
    execute_create_netcdf_label_start_time = datetime.datetime.now()
    # extract one hot encoding column
    labels = [col[0] for col in listOfDiacritizedCharacter]

    # convert unicode to string
    labels = [str(x) for x in labels]

    # convert array of string to array of char to be compatible with NETCDF
    # you will find strange values, but do not worry, it will exported correctly
    labels = netcdf_helpers.stringtochar(np.array(labels))

    global purified_labels
    purified_labels = []
    purified_labels = labels

    execute_create_netcdf_label_end_time = datetime.datetime.now()
    print "create_netcdf_label takes : ", execute_create_netcdf_label_end_time - execute_create_netcdf_label_start_time


def get_selected_letters_in_this_loop():
    execute_get_selected_letters_in_this_loop_start_time = datetime.datetime.now()

    global selected_letters_in_this_loop
    selected_letters_in_this_loop = []

    selected_letters_in_this_loop = listOfSelectedLettersAndSentences
    # selected_letters_in_this_loop = y
    execute_get_selected_letters_in_this_loop_end_time = datetime.datetime.now()
    print "executeChangedSQLQueries takes : ", \
        execute_get_selected_letters_in_this_loop_end_time - execute_get_selected_letters_in_this_loop_start_time


def create_netcdf_input():
    execute_create_netcdf_Input_Start_Time = datetime.datetime.now()
    searchCounter = 0

    global purified_netcdf_input
    purified_netcdf_input = []
    # Create Data of Input Variable
    for eachItem in range(0, len(selected_letters_in_this_loop)):
        yourLabel = selected_letters_in_this_loop[eachItem][0]
        un_diacritized_word = selected_letters_in_this_loop[eachItem][5]
        location = selected_letters_in_this_loop[eachItem][8]
        flag = True
        while flag:
            if listOfUnDiacritizedCharacter[searchCounter][1] == yourLabel:
                flag = False
                UnDiacritizedCharacterOneHotEncoding = map(int,
                                                           list(str(listOfUnDiacritizedCharacter[searchCounter][2])))
                try:
                    if location == 'first':
                        UnDiacritizedCharacterOneHotEncoding.append(1)
                        UnDiacritizedCharacterOneHotEncoding.append(0)
                        UnDiacritizedCharacterOneHotEncoding.append(0)
                    elif location == 'middle':
                        UnDiacritizedCharacterOneHotEncoding.append(0)
                        UnDiacritizedCharacterOneHotEncoding.append(1)
                        UnDiacritizedCharacterOneHotEncoding.append(0)
                    else:
                        UnDiacritizedCharacterOneHotEncoding.append(0)
                        UnDiacritizedCharacterOneHotEncoding.append(0)
                        UnDiacritizedCharacterOneHotEncoding.append(1)
                except:
                    raise Exception("error occurred in this loop")

                searchCounter = 0
                purified_netcdf_input.append(np.array(UnDiacritizedCharacterOneHotEncoding))
            else:
                searchCounter += 1

    execute_create_netcdf_Input_end_time = datetime.datetime.now()
    print "createNetCDFInput takes : ", execute_create_netcdf_Input_end_time - execute_create_netcdf_Input_Start_Time


def create_netcdf_seq_length():
    execute_create_netcdf_seq_length_start_time = datetime.datetime.now()
    letterCounterForEachSentence = 0
    global seq_lengths
    seq_lengths = []
    sentenceNumber = selected_letters_in_this_loop[0][3]

    # Create Data of SEQ Length Variable
    for eachItem in range(0, len(selected_letters_in_this_loop)):
        try:
            if selected_letters_in_this_loop[eachItem][3] == sentenceNumber:
                letterCounterForEachSentence += 1
            else:
                seq_lengths.append(letterCounterForEachSentence)
                prev = sentenceNumber
                sentenceNumber = selected_letters_in_this_loop[eachItem][3]

                letterCounterForEachSentence = 1
                if (int(sentenceNumber) - int(prev)) == 2:
                    x = 1
        except:
            x = 1

    seq_lengths.append(letterCounterForEachSentence)

    execute_create_netcdf_seq_length_end_time = datetime.datetime.now()
    print "createNetCDFSeqLength takes : ", execute_create_netcdf_seq_length_end_time - execute_create_netcdf_seq_length_start_time


def create_netcdf_target_classes():
    execute_create_netcdf_target_classes_start_time = datetime.datetime.now()
    letter = []
    diacritics = []
    searchCounter = 0
    targetClass = []
    beforeWhileLoop = datetime.datetime.now()
    for eachItem in range(0, len(selected_letters_in_this_loop)):
        yourLabel = selected_letters_in_this_loop[eachItem][7]
        OneHotTargetClassNotFound = True

        decomposed_letter = WordLetterProcessingHelperMethod.decompose_diac_char_into_char_and_diacritics(yourLabel)
        if len(decomposed_letter) == 2 and decomposed_letter[1] == u'ّ':
            # shadda and fatha
            decomposed_letter[1] = u'َّ'
            letter.append(decomposed_letter[0])
            diacritics.append(decomposed_letter[1])
            yourLabel = WordLetterProcessingHelperMethod.attach_diacritics_to_chars(letter, diacritics)[0]

        if len(decomposed_letter) == 2 and decomposed_letter[1] == u'ْ':
            # sukun case
            decomposed_letter[1] = u'ْ'
            letter.append(decomposed_letter[0])
            diacritics.append(decomposed_letter[1])
            yourLabel = decomposed_letter[0]

        elif len(decomposed_letter) == 2:
            letter.append(decomposed_letter[0])
            diacritics.append(decomposed_letter[1])
            yourLabel = decomposed_letter[1]
        elif len(decomposed_letter) > 3:
            x = 1
        elif len(decomposed_letter) > 2:
            letter.append(decomposed_letter[0])
            diacritics.append(decomposed_letter[1])
            yourLabel = decomposed_letter[1] + decomposed_letter[2]
            x = 1

        elif len(decomposed_letter) == 1:
            letter.append(decomposed_letter[0])
            diacritics.append(u'')
            yourLabel = decomposed_letter[0]

        while OneHotTargetClassNotFound:
            try:
                if listOfDiacritizedCharacter[searchCounter][1] == yourLabel:
                    OneHotTargetClassNotFound = False
                    targetClass.append(listOfDiacritizedCharacter[searchCounter][0])
                    searchCounter = 0
                else:
                    searchCounter += 1
            except:
                raise Exception("label not found")
    afterWhileLoop = datetime.datetime.now()
    print "While Loop takes : ", afterWhileLoop - beforeWhileLoop

    global purified_target_class
    purified_target_class = []

    purified_target_class = np.array(targetClass)
    execute_create_netcdf_target_class_end_time = datetime.datetime.now()
    print "createNetCDFTargetClasses takes : ", \
        execute_create_netcdf_target_class_end_time - execute_create_netcdf_target_classes_start_time


#  added due to error in running library
def create_seq_tags():
    execute_create_seq_tag_start_time = datetime.datetime.now()
    global seq_tag_sentences
    global max_seq_tag_length
    final = []
    final2 = []
    seq_tag_sentences = []
    counter = 1
    for eachItem in range(0, len(seq_lengths)):
        sentenceNumber = counter
        final = [int(i) for i in str(sentenceNumber)]
        number_of_zeros = max_seq_tag_length - len((final))
        sentenceNumber = str(sentenceNumber)
        for x in range (0, number_of_zeros):
            sentenceNumber = '0' + ',' + sentenceNumber
            final2.append(str(0))
        for x in range(0, len(final)):
            final2.append(str(final[x]))
        seq_tag_sentences.append((final2))
        final2 = []
        counter += 1

    seq_tag_sentences = (np.array(seq_tag_sentences))
   # for x in range (0, len(seq_tag_sentences)):
   #     final.append(seq_tag_sentences[x].split(','))

    #seq_tag_sentences = (np.array(final))
    #a = seq_tag_sentences[0].split(',')
    #v = np.fromstring(seq_tag_sentences, sep=',').reshape(-1, 10)
    #y = seq_tag_sentences[0]
    #x = np.reshape(seq_tag_sentences[0],(-1,2))
    execute_create_seq_tag_end_time = datetime.datetime.now()
    print "createSeqTags takes : ", execute_create_seq_tag_end_time - execute_create_seq_tag_start_time


def create_netcdf_file(dataset_type):

    outputFilename = dataset_type + "NCFile" + ".nc"

    # create a new .nc file
    dataset = netcdf_helpers.Dataset(outputFilename, 'w', format='NETCDF4')

    # create the dimensions
    dataset.createDimension('numTimesteps', len(purified_netcdf_input))
    dataset.createDimension('inputPattSize', len(purified_netcdf_input[0]))
    dataset.createDimension('numLabels', len(purified_labels))
    dataset.createDimension('maxLabelLength', len(purified_labels[0]))  # you get this value from the array 'labels'
    dataset.createDimension('numSeqs', len(seq_lengths))

    #  added due to error in running library
    dataset.createDimension('maxSeqTagLength', max_seq_tag_length)

    # create the variables
    netCDFLabels = dataset.createVariable('labels', 'S1', ('numLabels', 'maxLabelLength'))
    netCDFLabels[:] = purified_labels

    netCDFInput = dataset.createVariable('inputs', 'i4', ('numTimesteps', 'inputPattSize'))
    netCDFInput[:] = purified_netcdf_input

    netCDFSeq_lengths = dataset.createVariable('seqLengths', 'i4', 'numSeqs')
    netCDFSeq_lengths[:] = seq_lengths

    netCDFTargetClasses = dataset.createVariable('targetClasses', 'i4', 'numTimesteps')
    netCDFTargetClasses[:] = purified_target_class

    netCDFSeqTags = dataset.createVariable('seqTags', 'S1', ('numSeqs', 'maxSeqTagLength'))
    try:
        netCDFSeqTags[:] = seq_tag_sentences
    except:
        x = 1
    # write the data to disk
    print "writing data to", outputFilename
    dataset.close()


def try_using_query(letter):
    before = datetime.datetime.now()
    listOfSelectedLettersAndSentencesQuery = "select DiacritizedCharacterOneHotEncoding from " \
                                             "diaconehotencoding where DiacritizedCharacter= " + "'%s'" % letter

    cur.execute(listOfSelectedLettersAndSentencesQuery)
    global test1
    test1 = []
    test1 = cur.fetchall()
    after = datetime.datetime.now()
    print "modification takes : ", after - before
    return test1


if __name__ == "__main__":
    availableDataSetTypes = ['testing']
    columnNumberOf_SentenceNumber = 3

    create_mysql_connection()
    get_all_letters_of_corresponding_dataset_type(availableDataSetTypes[0])
    execute_unchanged_sql_queries()
    create_netcdf_label()

    startTime = datetime.datetime.now()

    get_selected_letters_in_this_loop()
    create_netcdf_input()
    create_netcdf_seq_length()

    create_netcdf_target_classes()
    create_seq_tags()
    create_netcdf_file(availableDataSetTypes[0])
    endTime = datetime.datetime.now()
    print "over all time ", endTime - startTime
