import DBHelperMethod
import DERCalculationHelperMethod
import DictionaryCorrection
import ExcelHelperMethod
import FathaCorrection
import RNNOPProcessingHelperMethod
import SukunCorrection
import WordLetterProcessingHelperMethod
import os
import glob
from copy import deepcopy
# actual = rnnop
extension = 'csv'
path = 'D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample\\'
path_error = "D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample\Errors\Book1.xls "
path_error_without_last_char = "D:\CurrenntRepo\CurrenntVS\CURRENNT\ArabicDiacritizationExample\Errors\Book2.xls "
current_row_1 = 0
current_row_2 = 0
Total_Error = 0
Total_Error_without_last_char = 0

if __name__ == "__main__":
    type = 'testing'
    list_of_sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by(type)
    os.chdir(path)
    result = [i for i in glob.glob('*.{}'.format(extension))]
    current_sentence_counter = 0

    if len(list_of_sentence_numbers) != len(result):
        raise Exception('Mismatch In Number Of Sentences')

    for file_name, sentence_number in zip(result, list_of_sentence_numbers):

        selected_sentence = DBHelperMethod.get_sentence_by(sentence_number)

        rnn_output = ExcelHelperMethod.read_rnn_op_csv_file(path + file_name)
        neurons_with_highest_probability = RNNOPProcessingHelperMethod.get_neurons_numbers_with_highest_output_value(rnn_output)

        list_of_available_diacritics = DBHelperMethod.get_all_diacritics()
        RNN_Predicted_diacritics = RNNOPProcessingHelperMethod.\
            deduce_from_rnn_op_predicted_chars(list_of_available_diacritics, neurons_with_highest_probability)

        IP_Undiacritized_Chars = DBHelperMethod.get_un_diacritized_chars_by(sentence_number, type)
        RNN_Predicted_chars = WordLetterProcessingHelperMethod.attach_diacritics_to_chars(IP_Undiacritized_Chars, RNN_Predicted_diacritics)

        RNN_Predicted_Chars_Count = WordLetterProcessingHelperMethod.get_chars_count_for_each_word_in_this(selected_sentence)
        RNN_Predicted_Chars_And_Its_Location = WordLetterProcessingHelperMethod.get_location_of_each_char(RNN_Predicted_chars, RNN_Predicted_Chars_Count)

        # Post Processing
        RNN_Predicted_Chars_After_Sukun = SukunCorrection.sukun_correction(deepcopy(RNN_Predicted_Chars_And_Its_Location))
        RNN_Predicted_Chars_After_Fatha = FathaCorrection.fatha_correction(deepcopy(RNN_Predicted_Chars_After_Sukun))
        RNN_Predicted_Chars_After_Dictionary = DictionaryCorrection.get_diac_version_with_smallest_dist(deepcopy(RNN_Predicted_Chars_After_Fatha), sentence_number)

        # Expected OP
        OP_Diac_Chars = DBHelperMethod.get_diacritized_chars_by(sentence_number, type)
        OP_Diac_Chars_Count = WordLetterProcessingHelperMethod.get_chars_count_for_each_word_in_this(
            selected_sentence)
        OP_Diac_Chars_And_Its_Location = WordLetterProcessingHelperMethod.get_location_of_each_char(
            OP_Diac_Chars, OP_Diac_Chars_Count)
        OP_Diac_Chars_After_Sukun = SukunCorrection.sukun_correction(
            deepcopy(OP_Diac_Chars_And_Its_Location))

        # DER Calculation
        error = DERCalculationHelperMethod.get_diacritization_error\
            (RNN_Predicted_Chars_After_Dictionary, OP_Diac_Chars_After_Sukun, selected_sentence)

        error_without_last_letter = DERCalculationHelperMethod.get_diacritization_error_without_counting_last_letter\
            (RNN_Predicted_Chars_After_Dictionary, OP_Diac_Chars_After_Sukun, selected_sentence)

        # write error in excel file
        excel_1 = current_row_1
        current_row_1 = ExcelHelperMethod.write_data_into_excel_file(error, selected_sentence, excel_1)
        Total_Error += len(error)
        print "Total Error: ", Total_Error


        excel_2 = current_row_2
        current_row_2 = ExcelHelperMethod.write_data_into_excel_file2(error_without_last_letter, selected_sentence, excel_2)
        Total_Error_without_last_char += len(error_without_last_letter)
        print "Total Error without Last Char: ", Total_Error_without_last_char
        print ""