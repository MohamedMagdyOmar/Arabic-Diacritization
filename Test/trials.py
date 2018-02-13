import DataPreprocessing as dp


if __name__ == "__main__":
    dp.establish_db_connection()
    dataset = dp.load_dataset_table("training")
    x = dp.load_nn_input_dataset_string(dataset[:, [0, 6]])
    y = dp.load_nn_labels_dataset_string(dataset[:, [0, 1]])

    sent_num, sen_len = dp.load_nn_seq_lengths(dataset[:, [3]])
    dp.prepare_for_padding(x, sent_num, sen_len)
    v = 1