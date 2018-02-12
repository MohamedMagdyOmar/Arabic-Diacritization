import netCDF4 as netcdf_helpers
import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime


max_seq_tag_length = 4


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
    start_time = datetime.datetime.now()

    list_of_selected_letters_and_sentences_query = "select UnDiacritizedCharacter, Diacritics, LetterType, " \
                                                   "SentenceNumber," \
                                                   "Word, UnDiacritizedWord, InputSequenceEncodedWords, " \
                                                   "TargetSequenceEncodedWords," \
                                                   "DiacritizedCharacter " \
                                                   "from ParsedDocument where LetterType=" + "'%s'" % type_of_dataset

    cur.execute(list_of_selected_letters_and_sentences_query)
    testing_data = cur.fetchall()

    end_time = datetime.datetime.now()
    print "get_all_letters_of_corresponding_dataset_type takes : ", end_time - start_time

    return testing_data


def get_input_and_target_data(type_of_dataset):
    start_time = datetime.datetime.now()

    list_of_input_character_query = "select InputSequenceEncodedWords from parseddocument " \
                                     "where LetterType=" + "'%s'" % type_of_dataset
    cur.execute(list_of_input_character_query)
    list_of_input_character = cur.fetchall()

    list_of_target_character_query = "select TargetSequenceEncodedWords from parseddocument " \
                                     "where LetterType=" + "'%s'" % type_of_dataset
    cur.execute(list_of_target_character_query)
    list_of_target_character = cur.fetchall()

    end_time = datetime.datetime.now()
    print "executeChangedSQLQueries takes : ", end_time - start_time

    return list_of_input_character, list_of_target_character


def create_netcdf_input(raw_data, input_data):
    space = u'0000000000000000'
    start_time = datetime.datetime.now()
    netcdf_input_class = []

    if len(raw_data) != len(input_data):
        raise 'Bug Appeared in create_netcdf_input'

    for index in range(0, len(input_data)):
        input_character = raw_data[index][0]
        un_diacritized_word = raw_data[index][5]

        try:
            if input_character == 'bos' or input_character == 'eos' or input_character == 'space':
                input_list = map(int, list(str(space)))
            else:
                input_list = map(int, list(str(input_data[index][0])))
        except:
            x = 1

        if input_character == 'space' or input_character == 'eos' or input_character == 'bos':
            input_list.append(0)
        elif un_diacritized_word != raw_data[(index + 1)][5]:
            input_list.append(1)
        else:
            input_list.append(0)

        netcdf_input_class.append(np.array(input_list))

    end_time = datetime.datetime.now()
    print "createNetCDFInputClasses takes : ", end_time - start_time

    return netcdf_input_class


def create_netcdf_seq_length(data):
    start_time = datetime.datetime.now()
    letter_counter_for_each_sentence = 0

    seq_lengths = []
    sentence_number = data[0][3]

    # Create Data of SEQ Length Variable
    for eachItem in range(0, len(data)):

        if data[eachItem][3] == sentence_number:
            letter_counter_for_each_sentence += 1
        else:
            seq_lengths.append(letter_counter_for_each_sentence)
            sentence_number = data[eachItem][3]
            letter_counter_for_each_sentence = 1

    seq_lengths.append(letter_counter_for_each_sentence)  # here

    end_time = datetime.datetime.now()
    print "createNetCDFSeqLength takes : ", end_time - start_time

    return seq_lengths


def create_netcdf_target_classes(target_data):
    space = u'0000000000000000'
    start_time = datetime.datetime.now()
    netcdf_target_class = []
    for each_item in target_data:
        if each_item[0] == 'bos' or each_item[0] == 'eos' or each_item[0] == 'space':
            netcdf_target_class.append(np.array(map(int, list(str(space)))))
        else:
            netcdf_target_class.append(np.array(map(int, list(str(each_item[0])))))

    end_time = datetime.datetime.now()
    print "createNetCDFTargetClasses takes : ", end_time - start_time

    return netcdf_target_class


#  added due to error in running library
def create_seq_tags(seq_lengths):
    start_time = datetime.datetime.now()

    global max_seq_tag_length

    final2 = []
    seq_tag_sentences = []
    counter = 1
    for eachItem in range(0, len(seq_lengths)):
        sentence_number = counter
        final = [int(i) for i in str(sentence_number)]
        number_of_zeros = max_seq_tag_length - len(final)
        sentence_number = str(sentence_number)
        for x in range(0, number_of_zeros):
            sentence_number = '0' + ',' + sentence_number
            final2.append(str(0))
        for x in range(0, len(final)):
            final2.append(str(final[x]))
        seq_tag_sentences.append(final2)
        final2 = []
        counter += 1

    seq_tag_sentences = (np.array(seq_tag_sentences))

    end_time = datetime.datetime.now()
    print "createSeqTags takes : ", end_time - start_time

    return seq_tag_sentences


def create_netcdf_file(dataset_type, netcdf_input, seq_lengths, target_class, seq_tag_sentences):
    output_file_name = dataset_type + "NCFile" + ".nc"

    # create a new .nc file
    dataset = netcdf_helpers.Dataset(output_file_name, 'w', format='NETCDF4')

    # create the dimensions
    dataset.createDimension('numSeqs', len(seq_lengths))
    dataset.createDimension('numTimesteps', len(netcdf_input))
    dataset.createDimension('inputPattSize', len(netcdf_input[0]))
    dataset.createDimension('targetPattSize', len(target_class[1]))

    #  added due to error in running library
    dataset.createDimension('maxSeqTagLength', max_seq_tag_length)

    # create the variables
    netCDFSeqTags = dataset.createVariable('seqTags', 'S1', ('numSeqs', 'maxSeqTagLength'))
    netCDFSeqTags[:] = seq_tag_sentences

    netCDFSeq_lengths = dataset.createVariable('seqLengths', 'i4', 'numSeqs')
    netCDFSeq_lengths[:] = seq_lengths

    netCDFInput = dataset.createVariable('inputs', 'i4', ('numTimesteps', 'inputPattSize'))
    netCDFInput[:] = netcdf_input

    netCDFTargetClasses = dataset.createVariable('targetPatterns', 'i4', ('numTimesteps', 'targetPattSize'))
    netCDFTargetClasses[:] = target_class

    # write the data to disk
    print "writing data to", output_file_name
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

    DataSetType = 'testing'
    create_mysql_connection()
    SelectedData = get_all_letters_of_corresponding_dataset_type(DataSetType)
    InputData, TargetData = get_input_and_target_data(DataSetType)
    startTime = datetime.datetime.now()
    NetCDFInput = create_netcdf_input(SelectedData, InputData)
    SEQLength = create_netcdf_seq_length(SelectedData)
    NetCDFTargetClass = create_netcdf_target_classes(TargetData)
    SEQTags = create_seq_tags(SEQLength)

    create_netcdf_file(DataSetType, NetCDFInput, SEQLength, NetCDFTargetClass, SEQTags)
    endTime = datetime.datetime.now()
    print "over all time ", endTime - startTime
