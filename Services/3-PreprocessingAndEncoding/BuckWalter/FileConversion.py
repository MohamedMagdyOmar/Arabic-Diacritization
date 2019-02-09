# -*- coding: utf-8 -*-

import os
import glob
import io
import re
path = 'C:\Users\Mohamed Magdy\Desktop\NewApproach\\atb3_v3_2\data\integrated'
extension = 'txt'

FileName = ''
list_of_words_in_selected_file = []
list_sent_of_list_of_words = []

selected_segment = []
doc = []
listOfPunctuationBySymbol = [':', '«', '»', '،', '؛', ',', '؟']


def read_txt_file(filename):
    path_of_file = 'C:\Users\Mohamed Magdy\Desktop\NewApproach\\atb3_v3_2\data\integrated\\' + filename

    with open(path_of_file) as f:
        content_of_selected_file = f.readlines()
        content_of_selected_file = [x.strip() for x in content_of_selected_file]

    global list_of_words_in_selected_file
    list_of_words_in_selected_file = []

    list_of_words_in_selected_file = [col for col in content_of_selected_file if col.startswith('s:')]


def select_segment_from_the_list_of_word():
    global selected_segment
    selected_segment = []

    global doc
    doc = []

    did_you_get_s0_letter_in_segment = False

    for col in list_of_words_in_selected_file:
        if not col.startswith('s:0') or did_you_get_s0_letter_in_segment == False:
            selected_segment.append(col)
            did_you_get_s0_letter_in_segment = True
        else:
            list_of_purified_segments = extract_buckwalter_string(selected_segment)
            doc.append(list_of_purified_segments)

            del list_of_purified_segments
            selected_segment = []
            selected_segment.append(col)

    write_new_doc(doc)


def extract_buckwalter_string(segment):
    initial_list_of_segment = []

    for each_row in segment:
        try:
            # letter_after_extraction = each_row.split('(', 1)[1].split(')')[0]
            result = re.search(']·\((.*)\)·OK', each_row)
            letter_after_extraction = result.group(1)
            x = 1
        except:
            letter_after_extraction = each_row.split('·', 1)[1].split('·')[1]

        if letter_after_extraction != "":
            initial_list_of_segment.append(letter_after_extraction)

        elif letter_after_extraction == "" and ((each_row.split('·', 1)[1].split('·')[0]) in listOfPunctuationBySymbol):
            initial_list_of_segment.append((each_row.split('·', 1)[1].split('·')[0]))

    return initial_list_of_segment


def write_new_doc(list_of_list_of_segments):
    with io.FileIO('C:\Users\Mohamed Magdy\Desktop\Test\\' + FileName, "w") as file:
        for each_segment in list_of_list_of_segments:
            for each_word in each_segment:
                if each_word not in listOfPunctuationBySymbol:
                    file.write(each_word + " ")
                else:
                    file.write("\n")
            file.write("\n")


if __name__ == "__main__":
    os.chdir(path)
    result = [i for i in glob.glob('*.{}'.format(extension))]

    for file_name in result:
        print file_name
        FileName = file_name
        read_txt_file(file_name)
        select_segment_from_the_list_of_word()
        x = 1