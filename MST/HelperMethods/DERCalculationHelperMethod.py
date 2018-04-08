from copy import deepcopy
import unicodedata
import WordLetterProcessingHelperMethod


class ErrorDetails:
    actual_letter = "",
    expected_letter = "",
    undiac_letter = ""
    error_location_in_word = 0,
    error_location_in_sentence = 0,
    act_word = ""
    exp_word = ""
    sentence_number = 0
    sentence = ''
    expected_diacritics = ""
    actual_diacritics = ""

    def __init__(self):
        self.actual_letter = "",
        self.expected_letter = "",
        self.undiac_letter = ""
        self.error_location_in_word = 0,
        self.error_location_in_sentence = 0,
        self.act_word = ""
        self.exp_word = ""
        self.sentence_number = 0
        self.sentence = ''
        self.expected_diacritics = ""
        self.actual_diacritics = ""


def get_diacritization_error(rnn_op_chars, expected_letters, sentence):
    list_of_object_error = []
    total_error = 0
    total_chars_including_un_diacritized_target_letter = 0

    number_of_diacritization_errors = 0
    letter_location = 0

    if len(rnn_op_chars) != len(expected_letters):
        raise Exception('bug appeared in "get_diacritization_error"')

    for actual_letter, expected_letter in zip(rnn_op_chars, expected_letters):

        error_object = ErrorDetails()
        decomposed_expected_letter = WordLetterProcessingHelperMethod.\
            decompose_diac_char_into_char_and_diacritics(expected_letter.letter)

        decomposed_actual_letter = WordLetterProcessingHelperMethod. \
            decompose_diac_char_into_char_and_diacritics(actual_letter.letter)

        letter_location += 1
        total_chars_including_un_diacritized_target_letter += 1
        # if = 1, this means that char is not diacritized, so do not consider it (as per paper)
        if len(decomposed_expected_letter) > 1:
            if actual_letter.letter != expected_letter.letter:
                error_object.actual_letter = actual_letter
                error_object.expected_letter = expected_letter
                error_object.error_location = letter_location
                error_object.word = get_word_that_has_error(letter_location, sentence)
                error_object.letter = decomposed_expected_letter[0]
                if len(decomposed_expected_letter) > 2:
                    error_object.expected_diacritics = decomposed_expected_letter[1] + decomposed_expected_letter[2]
                else:
                    error_object.expected_diacritics = decomposed_expected_letter[1]

                if len(decomposed_actual_letter) > 2:
                    error_object.actual_diacritics = decomposed_actual_letter[1] + decomposed_actual_letter[2]

                elif len(decomposed_actual_letter) > 1:
                    error_object.actual_diacritics = decomposed_actual_letter[1]

                list_of_object_error.append(deepcopy(error_object))
                number_of_diacritization_errors += 1

    total_error += number_of_diacritization_errors

    print('total error in this sentence', number_of_diacritization_errors)

    return list_of_object_error


def get_diacritization_error_version_2(master_object, sentence_number, sentence):
    list_of_object_error = []
    number_of_diacritization_errors = 0

    for each_object in master_object:

        error_object = ErrorDetails()
        # if = 1, this means that char is not diacritized, so do not consider it (as per paper)
        if each_object.exp_diac != '' and each_object.exp_diac != each_object.rnn_diac:
                error_object.actual_letter = each_object.rnn_diac_char
                error_object.expected_letter = each_object.exp_diac_char
                error_object.undiac_letter = each_object.undiac_char
                error_object.error_location_in_word = each_object.location_in_word
                error_object.error_location_in_sentence = each_object.location_in_sent
                error_object.exp_word = each_object.exp_diac_word
                error_object.act_word = each_object.rnn_diac_word

                error_object.expected_diacritics = each_object.exp_diac
                error_object.actual_diacritics = each_object.rnn_diac
                error_object.sentence_number = sentence_number
                error_object.sentence = sentence

                list_of_object_error.append(deepcopy(error_object))
                number_of_diacritization_errors += 1

    print('total error in this sentence', number_of_diacritization_errors)

    return list_of_object_error


def get_diacritization_error_without_counting_last_letter(actual_letters, expected_letters, sentence):
    list_of_object_error = []
    total_error_without_last_letter = 0

    number_of_diacritization_errors = 0
    letter_location = 0

    if len(actual_letters) != len(expected_letters):
        raise Exception('bug appeared in "get_diacritization_error_without_counting_last_letter"')

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

    print('total error in this sentence (without Last Letter):', number_of_diacritization_errors)

    return list_of_object_error


def get_diacritization_error_without_counting_last_letter_version_2(master_object, sentence_number, sentence):
    list_of_object_error = []
    number_of_diacritization_errors = 0

    for each_object in master_object:
        error_object = ErrorDetails()
        if each_object.location_in_word != 'last':
                    if each_object.exp_diac != '' and each_object.exp_diac != each_object.rnn_diac:
                        error_object.actual_letter = each_object.rnn_diac_char
                        error_object.expected_letter = each_object.exp_diac_char
                        error_object.error_location = each_object.location_in_word
                        error_object.exp_word = each_object.exp_diac_word
                        error_object.act_word = each_object.rnn_diac_word
                        error_object.letter = each_object.undiac_char
                        error_object.expected_diacritics = each_object.exp_diac
                        error_object.actual_diacritics = each_object.rnn_diac
                        error_object.sentence_number = sentence_number
                        error_object.sentence = sentence

                        list_of_object_error.append(deepcopy(error_object))
                        number_of_diacritization_errors += 1

    print('total error in this sentence (without Last Letter):', number_of_diacritization_errors)

    return list_of_object_error


def get_diacritization_error_without_sentence(rnn_op_chars, expected_letters):
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

    print('total error in this sentence', number_of_diacritization_errors)

    return list_of_object_error


def get_diacritization_error_without_counting_last_letter_without_sent(actual_letters, expected_letters):
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

    print('total error in this sentence (without Last Letter):', number_of_diacritization_errors)

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
