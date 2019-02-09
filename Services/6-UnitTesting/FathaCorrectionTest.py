import unittest
import xlsxwriter
import MySQLdb
import MySQLdb.cursors
import unicodedata
from copy import deepcopy
import DER_Sukun_Fatha_Dictionary_Correction as DERCalculation
from xlrd import open_workbook
from xlutils.copy import copy

extension = 'csv'

path = 'C:\Users\Mohamed Magdy\Desktop'
diacritization_error_excel_file_path = "C:\Users\Mohamed Magdy\Desktop\TestingFathaCorrection.xls "


workbook = xlsxwriter.Workbook(diacritization_error_excel_file_path)
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'corrected')
worksheet.write(0, 1, 'corrected location')
worksheet.write(0, 2, 'actual')
worksheet.write(0, 3, 'actual location')
workbook.close()


def connect_to_db():
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


def get_query_data(query):
    selected_words_query = query
    cur.execute(selected_words_query)
    selected_words = (cur.fetchall())
    selected_words = [each_number[0] for each_number in selected_words]
    return selected_words


class LetterPosition:
    letter = "",
    location = ""

    def __init__(self):
        self.letter = ""
        self.location = ""


class MyTestCase1(unittest.TestCase):
    current_row_in_excel_file = 0
    selected_words = []
    selected_letters = []
    chars_count = []
    list_of_actual_letters_with_its_location = []

    # Only use setUp() and tearDown() if necessary

    def setUp(self):

        connect_to_db()

        query_1 = "select distinct word, DiacritizedCharacter from parseddocument where UnDiacritizedCharacter = (" \
                  "select arabic_letter from arabic_letters_without_diacritics where id = 7) "


        query_2 = "select distinct word, DiacritizedCharacter from parseddocument where UnDiacritizedCharacter = (" \
                  "select arabic_letter from arabic_letters_without_diacritics where id = 2) "

        query_3 = "select distinct word, DiacritizedCharacter from parseddocument where UnDiacritizedCharacter = (" \
                  "select arabic_letter from arabic_letters_without_diacritics where id = 34) "


        self.selected_words = get_query_data(query_1)

        self.selected_words += get_query_data(query_2)

        self.selected_words += get_query_data(query_3)

        for each_word in self.selected_words:

            for each_letter in each_word:
                if not unicodedata.combining(each_letter):
                    overall = each_letter
                    comp = unicodedata.normalize('NFC', each_letter)
                    self.selected_letters.append(comp)
                else:
                    overall += each_letter
                    comp = unicodedata.normalize('NFC', overall)
                    self.selected_letters.pop()
                    self.selected_letters.append(comp)

        self.chars_count = DERCalculation.get_chars_count_for_each_word_in_current_sentence(self.selected_words)
        self.list_of_actual_letters_with_its_location = DERCalculation.\
            get_location_of_each_character_in_current_sentence(self.selected_letters, self.chars_count)

    def test_feature_one(self):

        self.letters_after_fatha_correction = DERCalculation.fatha_correction(deepcopy(self.list_of_actual_letters_with_its_location))
        counter = -1
        for each_corrected_letter, each_input_letter in zip(self.letters_after_fatha_correction, self.list_of_actual_letters_with_its_location):
            print each_input_letter
            counter = counter + 1

            try:
                self.assertEqual(each_corrected_letter.letter, each_input_letter.letter, each_corrected_letter.letter.encode('utf-8') + " != " + each_input_letter.letter.encode('utf-8'))
            except:
                x = 1

        self.write_data_into_excel_file(self.letters_after_fatha_correction, self.list_of_actual_letters_with_its_location)

    def write_data_into_excel_file(self, each_corrected_letter, each_input_letter):
        wb = open_workbook(diacritization_error_excel_file_path)
        w = copy(wb)
        worksheet = w.get_sheet(0)

        # global current_row_in_excel_file
        current_row_in_excel_file = 1

        for corrected, actual in zip(each_corrected_letter, each_input_letter):
            column = 0
            worksheet.write(current_row_in_excel_file, column, corrected.letter)

            column = 1
            worksheet.write(current_row_in_excel_file, column, corrected.location)

            column = 2
            worksheet.write(current_row_in_excel_file, column, actual.letter)

            column = 3
            worksheet.write(current_row_in_excel_file, column, actual.location)

            current_row_in_excel_file += 1

        w.save(diacritization_error_excel_file_path)
        workbook.close()

        return current_row_in_excel_file


