import WordLetterProcessingHelperMethod
import DBHelperMethod
import SukunCorrection


def get_diac_version_with_smallest_dist(list_of_objects, sentence_number):

    list_of_actual_words_after_dictionary_correction = []

    diacritized_rnn_op_words = WordLetterProcessingHelperMethod.reform_word_from(list_of_objects)
    undiacritized_words = DBHelperMethod.get_un_diacritized_words_from(sentence_number, 'testing')

    if len(diacritized_rnn_op_words) != len(diacritized_rnn_op_words):
        raise Exception("error appeared in get_diac_version_with_smallest_dist")

    for each_corrected_word, each_un_diacritized_word in zip(diacritized_rnn_op_words, undiacritized_words):

        minimum_error = 100000000
        dictionary_diacritized_words = DBHelperMethod.\
            get_dictionary_all_diacritized_version_of(each_un_diacritized_word)

        if len(dictionary_diacritized_words) == 0:
            dictionary_diacritized_words.append(each_corrected_word)

        dictionary_diacritized_words_after_sukun_correction = SukunCorrection.\
            sukun_correction_for_list_of_words(dictionary_diacritized_words)

        if do_we_need_to_search_in_dictionary(dictionary_diacritized_words_after_sukun_correction, each_corrected_word):

            for each_word in dictionary_diacritized_words_after_sukun_correction:
                error_count = 0

                decomposed_dic_word = WordLetterProcessingHelperMethod.decompose_word_into_letters(each_word)
                decomposed_act_word = WordLetterProcessingHelperMethod.decompose_word_into_letters(each_corrected_word)

                norm_dic_word = WordLetterProcessingHelperMethod.normalize(decomposed_dic_word)
                norm_act_word = WordLetterProcessingHelperMethod.normalize(decomposed_act_word)

                for each_diacritized_version_letter, each_current_word_letter in zip(norm_dic_word, norm_act_word):

                    if (len(each_diacritized_version_letter) - len(each_current_word_letter) == 1) or (
                            (len(each_diacritized_version_letter) - len(each_current_word_letter) == -1)):
                        error_count += 1

                    elif (len(each_diacritized_version_letter) - len(each_current_word_letter) == 2) or (
                            (len(each_diacritized_version_letter) - len(each_current_word_letter) == -2)):
                        error_count += 2

                    else:
                        for each_item_in_diacritized_version, each_item_in_current_word in \
                                zip(each_diacritized_version_letter, each_current_word_letter):
                            if each_item_in_diacritized_version != each_item_in_current_word:
                                error_count += 1

                if error_count < minimum_error:
                    minimum_error = error_count

                    selected_dictionary_word = each_word

            list_of_actual_words_after_dictionary_correction.append(selected_dictionary_word)
        else:
            list_of_actual_words_after_dictionary_correction.append(each_corrected_word)

    chars_after_dic_correction = WordLetterProcessingHelperMethod.convert_list_of_words_to_list_of_chars(list_of_actual_words_after_dictionary_correction)
    if len(list_of_objects) != len(chars_after_dic_correction):
        raise Exception("Error Happened Here")

    for x in range(0, len(list_of_objects)):
        list_of_objects[x].letter = chars_after_dic_correction[x]

    return list_of_objects


def do_we_need_to_search_in_dictionary(dictionary, word):

    for each_word in dictionary:
        decomposed_dict = WordLetterProcessingHelperMethod.decompose_word_into_letters(each_word)
        decomposed_act = WordLetterProcessingHelperMethod.decompose_word_into_letters(word)

        norm_dict = WordLetterProcessingHelperMethod.normalize(decomposed_dict)
        norm_act = WordLetterProcessingHelperMethod.normalize(decomposed_act)

        if len(norm_dict) != len(norm_act):
            raise ValueError("Bug Found In 'do_we_need_to_search_in_dictionary'")

        if sorted(norm_dict) == sorted(norm_act):
            return False

    for each_word in dictionary:
        decomposed_dict = WordLetterProcessingHelperMethod.decompose_word_into_letters(each_word)
        decomposed_act = WordLetterProcessingHelperMethod.decompose_word_into_letters(word)

        norm_dict = WordLetterProcessingHelperMethod.normalize(decomposed_dict)
        norm_act = WordLetterProcessingHelperMethod.normalize(decomposed_act)

        for x in range(0, len(norm_act)):
            # compare letters before last letter
            if x < (len(norm_act) - 1):
                if norm_dict[x] != norm_act[x]:
                    # so diff is in first or middle letters
                    break
            else:
                # so diff is in last letter so ignore it
                return False

    return True
