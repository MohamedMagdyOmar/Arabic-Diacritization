import numpy as np

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)


def encodeMyCharacterWith2Parameters(list_of_un_diacritized_character, list_of_diacritized_character):

    n__un_diacritized_classes = len(list_of_un_diacritized_character)
    n__diacritized_classes = len(list_of_diacritized_character)

    one_hot_list__for_un_diacritized_characters = one_hot_encode(list_of_un_diacritized_character, n__un_diacritized_classes)
    one_hot_list__for_diacritized_characters = one_hot_encode(list_of_diacritized_character, n__diacritized_classes)

    return one_hot_list__for_un_diacritized_characters,one_hot_list__for_diacritized_characters;


def encodeMyCharacterWith1Parameter(list_of_characters):

    n__character_classes = len(list_of_characters)

    one_hot_list__of_characters = one_hot_encode(list_of_characters, n__character_classes)

    return one_hot_list__of_characters
