from copy import deepcopy
import unicodedata
import WordLetterProcessingHelperMethod


class ErrorDetails:
    actual_letter = "",
    expected_letter = "",
    error_location = 0,
    word = ""

    def __init__(self):
        self.actual_letter = ""
        self.expected_letter = ""
        self.error_location = 0
        self.word = ""


def get_diacritization_error(rnn_op_chars, expected_letters, sentence):
    list_of_object_error = []
    total_error = 0
    total_chars_including_un_diacritized_target_letter = 0

    number_of_diacritization_errors = 0
    letter_location = 0

    if len(rnn_op_chars) != len(expected_letters):
        raise ValueError('bug appeared in "get_diacritization_error"')

    for actual_letter, expected_letter in zip(rnn_op_chars, expected_letters):

        error_object = ErrorDetails()
        decomposed_expected_letter = WordLetterProcessingHelperMethod.\
            decompose_diac_char_into_char_and_diacritics(expected_letter.letter)

        letter_location += 1
        total_chars_including_un_diacritized_target_letter += 1
        # if = 1, this means that char is not diacritized, so do not consider it (as per paper)
        if len(decomposed_expected_letter) > 1:
            if actual_letter.letter != expected_letter.letter:
                error_object.actual_letter = actual_letter
                error_object.expected_letter = expected_letter
                error_object.error_location = letter_location
                error_object.word = get_word_that_has_error(letter_location, sentence)

                list_of_object_error.append(deepcopy(error_object))
                number_of_diacritization_errors += 1

    total_error += number_of_diacritization_errors

    print 'total error in this sentence', number_of_diacritization_errors

    return list_of_object_error


def get_diacritization_error_without_counting_last_letter(actual_letters, expected_letters, sentence):
    list_of_object_error = []
    total_error_without_last_letter = 0

    number_of_diacritization_errors = 0
    letter_location = 0

    if len(actual_letters) != len(expected_letters):
        raise ValueError('bug appeared in "get_diacritization_error_without_counting_last_letter"')

    for actual_letter, expected_letter in zip(actual_letters, expected_letters):
        error_object = ErrorDetails()

        letter_location += 1
        if actual_letter.location != 'last' and expected_letter.location != 'last':
            decomposed_expected_letter = WordLetterProcessingHelperMethod.\
                decompose_diac_char_into_char_and_diacritics(expected_letter.letter)

            if actual_letter.location == expected_letter.location:

                if len(decomposed_expected_letter) > 1:
                    if actual_letter.letter != expected_letter.letter:
                        error_object.actual_letter = actual_letter
                        error_object.expected_letter = expected_letter
                        error_object.error_location = letter_location
                        error_object.word = get_word_that_has_error(letter_location, sentence)

                        list_of_object_error.append(deepcopy(error_object))
                        number_of_diacritization_errors += 1
            else:
                raise ValueError('bug appeared in "get_diacritization_error_without_counting_last_letter"')

    total_error_without_last_letter += number_of_diacritization_errors

    print 'total error in this sentence (without Last Letter):', number_of_diacritization_errors

    return list_of_object_error


def get_word_that_has_error(error_location, sentence):
    counter = 0
    for each_word in sentence:
        each_word = unicodedata.normalize('NFD', each_word)
        for each_char in each_word:
            if not unicodedata.combining(each_char):
                counter += 1
                if error_location == counter:
                    return each_word
