# -*- coding: utf-8 -*-

import unittest
import xlsxwriter
from copy import deepcopy
from xlrd import open_workbook
from xlutils.copy import copy
import ExcelHelperMethod
import WordLetterProcessingHelperMethod
import FathaCorrection

extension = 'csv'

# select  distinct word from parseddocument where lettertype='testing' and diacritics = 'ْ'
path = 'C:\Users\Mohamed Magdy\Desktop\\'
diacritization_error_excel_file_path = "C:\Users\Mohamed Magdy\Desktop\esttest.xls "


workbook = xlsxwriter.Workbook(diacritization_error_excel_file_path)
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'corrected')
worksheet.write(0, 1, 'corrected location')
worksheet.write(0, 2, 'actual')
worksheet.write(0, 3, 'actual location')
workbook.close()


class TehMarbotaTest(unittest.TestCase):

    def setUp(self):
        self.file_name = 'alfmaksora.csv'
        self.tehmarbota_words = ExcelHelperMethod.read_csv_file(path + self.file_name)
        self.chars_count = WordLetterProcessingHelperMethod.get_chars_count_for_each_word_in_this(deepcopy(self.tehmarbota_words))
        self.Chars = WordLetterProcessingHelperMethod.convert_list_of_words_to_list_of_chars(deepcopy(self.tehmarbota_words))

        self.Chars_And_Its_Location = WordLetterProcessingHelperMethod.get_location_of_each_char(
            self.Chars, self.chars_count)

    def test_feature_one(self):
        self.Chars_After_tehmarbota_correction = FathaCorrection.fatha_correction(
            deepcopy(self.Chars_And_Its_Location))

        self.write_data_into_excel_file(self.Chars_After_tehmarbota_correction, self.Chars_And_Its_Location)

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


