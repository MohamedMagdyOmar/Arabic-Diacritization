import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    xl = pd.ExcelFile("Book1.xls")
    sheet_1 = pd.read_excel("Book1.xls", sheetname=0)

    expected_diacritics = list(sheet_1['expected diacritics english'])
    exp_diacritics = set(expected_diacritics)

    actual_diacritics = list(sheet_1['actual diacritics english'])
    act_diacritics = set(actual_diacritics)

    labels = list(exp_diacritics)
    cm_numpy = confusion_matrix(expected_diacritics, actual_diacritics, labels=labels)
    print_confusion_matrix(cm_numpy, labels)
    print(cm_numpy)

    plt.show()
    y = 1

