import DataPreprocessing as dp


def load_training_data():

    training_dataset = dp.load_dataset_by_type("training")

    x = dp.load_nn_input_dataset_string(training_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(training_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(training_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.pad_sentences(x, sen_len, 4, 10)

    return sentences_padded, y, vocabulary, vocabulary_inv


def load_testing_data():

    testing_dataset = dp.load_dataset_by_type("testing")

    x = dp.load_nn_input_dataset_string(testing_dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(testing_dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(testing_dataset[:, [3]])
    sentences_padded, vocabulary, vocabulary_inv = dp.pad_sentences(x, sen_len, 4, 10)

    return sentences_padded, y, vocabulary, vocabulary_inv


if __name__ == "__main__":
    dp.establish_db_connection()
    X_train, y_train, vocabulary_train, vocabulary_inv_train = load_training_data()
    X_test, y_test, vocabulary_test, vocabulary_inv_test = load_testing_data()
    v = 1