import MySQLdb
import MySQLdb.cursors


def establish_db_connection():
    db = MySQLdb.connect(host="127.0.0.1",  # your host, usually localhost
                         user="root",  # your username
                         passwd="Islammega88",  # your password
                         db="mstdb",  # name of the data base
                         cursorclass=MySQLdb.cursors.SSCursor,
                         use_unicode=True,
                         charset="utf8",
                         init_command='SET NAMES UTF8')
    global cur
    cur = db.cursor()


def create_sent_from_list_of_chars(x, sent_len):
    start_range = 0
    end_range = 0
    extracted_sent = []
    for each_sent in range(0, len(sent_len)):
        end_range = sent_len[each_sent] + end_range
        extracted_sent.append(x[start_range: end_range])
        start_range = end_range

    return extracted_sent


def create_window_of_chars(input, T):
    window_of_chars = []
    counter = 1

    for selected_window in range(0, len(input), T):
            new_list = list(input[selected_window: counter * T])
            window_of_chars.append(new_list)
            counter += 1

    return window_of_chars


if __name__ == "__main__":

    establish_db_connection()

