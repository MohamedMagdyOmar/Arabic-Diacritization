# -*- coding: utf-8 -*-

import unicodedata


def sukun_correction(list_of_objects_of_chars_and_its_location):
    list_of_objects = []

    for each_object in list_of_objects_of_chars_and_its_location:
        each_object.letter = remove_sukun_diacritics_if_exists(each_object.letter)
        list_of_objects.append(each_object)

    return list_of_objects


def sukun_correction_for_list_of_words(list_of_words):
    words_without_sukun = []
    overall = ""

    for each_word in list_of_words:
        for each_char in each_word:
            spaChar = unicodedata.normalize('NFC', each_char)
            if u'ْ' in spaChar:
                    if not unicodedata.combining(spaChar):
                        overall += spaChar
                        words_without_sukun.append(unicodedata.normalize('NFC', overall))
            else:
                overall += spaChar
        words_without_sukun.append(unicodedata.normalize('NFC', overall))
        overall = ""

    return words_without_sukun


def remove_sukun_diacritics_if_exists(char):

    normalized_char = unicodedata.normalize('NFC', char)

    if u'ْ' in normalized_char:
        for c in normalized_char:
            if not unicodedata.combining(c):
                return unicodedata.normalize('NFC', c)
    else:
        return char
