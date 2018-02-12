import csv
import xlsxwriter
from xlrd import open_workbook
from xlutils.copy import copy
import WordLetterProcessingHelperMethod

extension = 'csv'

diacritization_error_excel_file_path = \
    "D:\Repos\\results\\41\\ff\\Book1.xls "

diacritization_error_without_last_letter_excel_file_path = \
    "D:\Repos\\results\\41\\ff\\Book2.xls "

workbook = xlsxwriter.Workbook(diacritization_error_excel_file_path)
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'RNN OP')
worksheet.write(0, 1, 'Required Target')
worksheet.write(0, 2, 'Error Location')
worksheet.write(0, 3, 'word contain error')
worksheet.write(0, 4, 'location')
worksheet.write(0, 5, 'sentence')
worksheet.write(0, 6, 'value')
workbook.close()

workbook2 = xlsxwriter.Workbook(diacritization_error_without_last_letter_excel_file_path)
worksheet2 = workbook2.add_worksheet()
worksheet2.write(0, 0, 'RNN OP')
worksheet2.write(0, 1, 'Required Target')
worksheet2.write(0, 2, 'Error Location')
worksheet2.write(0, 3, 'word contain error')
worksheet2.write(0, 4, 'location')
worksheet2.write(0, 5, 'sentence')
worksheet2.write(0, 6, 'value')
workbook2.close()


def read_rnn_op_csv_file(csv_complete_path):

    rnn_op = []

    with open(csv_complete_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:
            rnn_op.append(map(float, row))

        return rnn_op


def write_data_into_excel_file(errors, current_sentence, current_row_in_excel_file):
        wb = open_workbook(diacritization_error_excel_file_path)
        w = copy(wb)
        worksheet = w.get_sheet(0)

        all_sentence = ''
        for each_word in current_sentence:
            all_sentence += each_word + ' '

        current_row_in_excel_file += 1
        column = 0

        for each_object in errors:
            worksheet.write(current_row_in_excel_file, column, each_object.actual_letter.letter)

            column = 1
            worksheet.write(current_row_in_excel_file, column, each_object.expected_letter.letter)

            column = 2
            worksheet.write(current_row_in_excel_file, column, each_object.error_location)

            column = 3
            worksheet.write(current_row_in_excel_file, column, each_object.word)

            column = 4
            worksheet.write(current_row_in_excel_file, column, each_object.actual_letter.location)

            column = 5
            worksheet.write(current_row_in_excel_file, column, all_sentence)

            column = 6
            worksheet.write(current_row_in_excel_file, column, each_object.actual_letter.value)

            current_row_in_excel_file += 1
            column = 0

        current_row_in_excel_file += 1

        w.save(diacritization_error_excel_file_path)
        workbook.close()

        return current_row_in_excel_file


def write_data_into_excel_file2(errors, current_sentence, current_row_in_excel_file):
    wb = open_workbook(diacritization_error_without_last_letter_excel_file_path)
    w = copy(wb)
    worksheet2 = w.get_sheet(0)

    all_sentence = ''
    for each_word in current_sentence:
        all_sentence += each_word + ' '

    current_row_in_excel_file += 1
    column = 0

    for each_object in errors:
        worksheet2.write(current_row_in_excel_file, column, each_object.actual_letter.letter)

        column = 1
        worksheet2.write(current_row_in_excel_file, column, each_object.expected_letter.letter)

        column = 2
        worksheet2.write(current_row_in_excel_file, column, each_object.error_location)

        column = 3
        worksheet2.write(current_row_in_excel_file, column, each_object.word)

        column = 4
        worksheet2.write(current_row_in_excel_file, column, each_object.actual_letter.location)

        column = 5
        worksheet2.write(current_row_in_excel_file, column, all_sentence)

        column = 6
        worksheet2.write(current_row_in_excel_file, column, each_object.actual_letter.value)

        current_row_in_excel_file += 1
        column = 0

    current_row_in_excel_file += 1

    w.save(diacritization_error_without_last_letter_excel_file_path)
    workbook.close()

    return current_row_in_excel_file


def read_csv_file(csv_complete_path):
    csv_op = []

    with open(csv_complete_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:
            csv_op.append(unicode(row[0], "utf-8"))

        return csv_op