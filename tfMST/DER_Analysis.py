import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import numpy
import itertools
# A -> 0 -> RNN OP
# B -> 1 -> Required Target
# C -> 2 -> Error Location
# D -> 3 -> word contain error
# E -> 4 -> location
# F -> 5 -> Sentence
# G -> 6 -> sentence number
# H -> 7 -> letter
# I -> 8 -> expected diacritics
# J -> 9 -> actual diacritics


class diacritics_position_error_count:
    diacritic = "",
    position = "",
    error_count = ""

    def __init__(self):
        self.diacritic = ""
        self.position = ""
        self.error_count = ""


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('expected (correct) label')
    plt.xlabel('Predicted (rnn op) label')
    return fig


def calculate_num_errors_in_each_pos(df):
    locations = list(sheet1_df['location'])
    loc_and_number_of_error = Counter(locations)
    number_of_error_in_first_letter = loc_and_number_of_error['first']
    number_of_error_in_middle_letter = loc_and_number_of_error['middle']
    number_of_error_in_last_letter = loc_and_number_of_error['last']

    list1, list2 = ['number_of_error_in_first_letter',
                    'number_of_error_in_middle_letter',
                    'number_of_error_in_last_letter'], \
                   [number_of_error_in_first_letter,
                    number_of_error_in_middle_letter,
                    number_of_error_in_last_letter]

    result = dict(zip(list1, list2))
    return result


def calculate_num_errors_in_each_diac(df):
    expected_diac = list(df['expected diacritics'])
    actual_diac = list(df['actual diacritics'])

    distinct_expected_diac = set(expected_diac)
    actual_diac.extend(distinct_expected_diac)
    distinct_diacs = set(actual_diac)
    cols = (df.filter(items=['expected diacritics', 'location']))
    f = cols.get_values()

    result = []

    for each_diac in distinct_diacs:
        diacs = f[:, 0]
        idx_of_selected_diacs = diacs == each_diac
        selected_pair = f[idx_of_selected_diacs]
        positions = selected_pair[:, 1]
        if len(positions) != 0:
            b = Counter(positions)
            for each_key in b:
                new_instance = diacritics_position_error_count()
                new_instance.diacritic = each_diac
                new_instance.position = each_key
                new_instance.error_count = b[each_key]
                result.append(new_instance)

    return result


def calculate_num_errors_in_each_word(df):

    f = (df.filter(items=['word contain error', 'location'])).get_values()
    all_words = f[:, 0]
    all_locations = f[:, 1]

    has_one_error = 0
    has_one_error_and_include_last = 0
    has_one_error_and_not_include_last = 0

    has_two_error = 0
    has_two_error_and_include_last = 0
    has_two_error_and_not_include_last = 0

    has_three_or_more_error = 0
    has_three_or_more_error_and_include_last = 0
    has_three_or_more_error_and_not_include_last = 0

    start_range = 0

    #values = set(map(lambda x: x[1], list(f)))
    #newlist = [[y[0] for y in list(f) if y[1] == x] for x in values]

    for (key, group) in itertools.groupby(all_words):

        groups = list(group)

        end_range = start_range + len(groups)

        if len(groups) == 1:

            has_one_error += 1
            selected_locations = all_locations[start_range: end_range]
            if 'last' in selected_locations:
                has_one_error_and_include_last += 1
            else:
                has_one_error_and_not_include_last += 1

        elif len(groups) == 2:
            has_two_error += 1

            selected_locations = all_locations[start_range: end_range]
            if 'last' in selected_locations:
                has_two_error_and_include_last += 1
            else:
                has_two_error_and_not_include_last += 1

        elif len(groups) >= 3:
            has_three_or_more_error += 1
            selected_locations = all_locations[start_range: end_range]
            if 'last' in selected_locations:
                has_three_or_more_error_and_include_last += 1
            else:
                has_three_or_more_error_and_not_include_last += 1

        start_range = end_range

    list1, list2 = ['number_of_words_has_one_error',
                    'number_of_words_has_one_error_and_include_last',
                    'number_of_words_has_one_error_and_not_include_last',
                    'number_of_words_has_2_error',
                    'number_of_words_has_2_error_and_include_last',
                    'number_of_words_has_2_error_and_not_include_last',
                    'number_of_words_has_3_error',
                    'number_of_words_has_3_error_and_include_last',
                    'number_of_words_has_3_error_and_not_include_last'], \
                   [has_one_error,
                    has_one_error_and_include_last,
                    has_one_error_and_not_include_last,
                    has_two_error,
                    has_two_error_and_include_last,
                    has_two_error_and_not_include_last,
                    has_three_or_more_error,
                    has_three_or_more_error_and_include_last,
                    has_three_or_more_error_and_not_include_last]

    result = dict(zip(list1, list2))
    return result


def draw_confusion_matrix(df):
    expected_diacritics = list(df['expected diacritics english'])
    exp_diacritics = set(expected_diacritics)

    actual_diacritics = list(df['actual diacritics english'])

    labels = list(exp_diacritics)
    cm_numpy = confusion_matrix(expected_diacritics, actual_diacritics, labels=labels)
    print_confusion_matrix(cm_numpy, labels)

    plt.show()


if __name__ == "__main__":

    xl = pd.ExcelFile("Book1.xls")
    sheet1_df = pd.read_excel("Book1.xls", sheetname=0)

    calculate_num_errors_in_each_pos(sheet1_df)
    calculate_num_errors_in_each_diac(sheet1_df)
    calculate_num_errors_in_each_word(sheet1_df)
    draw_confusion_matrix(sheet1_df)


