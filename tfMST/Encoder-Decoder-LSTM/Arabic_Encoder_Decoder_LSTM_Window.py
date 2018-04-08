from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import DBHelperMethod
import data_helper as dp
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 64  # Batch size for training.
epochs = 35  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

input_characters = set()
target_characters = set()

dp.establish_db_connection()

training_dataset = DBHelperMethod.load_dataset_by_type("training")
all_training_sentences_numbers = training_dataset[:, 3]

testing_dataset = DBHelperMethod.load_dataset_by_type("testing")
all_testing_sentences_numbers = training_dataset[:, 3]

list_of_training_sentence_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("training")
list_of_testing_sentences_numbers = DBHelperMethod.get_list_of_sentence_numbers_by("testing")

input_texts = training_dataset[:, 0]
target_texts = dp.load_nn_labels(training_dataset[:, [0, 1]])

new_input_text = []
new_target_text = []
window_size = 13

start_range = 0
end_range = 0
window_input_text = []
window_target_text = []

start_time = datetime.datetime.now()
for each_sentence_number in list_of_training_sentence_numbers:

    number_of_occurrence = (list(all_training_sentences_numbers)).count(str(each_sentence_number))
    end_range = start_range + int(number_of_occurrence)
    extracted_input_text = input_texts[start_range: end_range]
    extracted_target_text = target_texts[start_range: end_range]
    start_range = end_range

    division_result = len(extracted_input_text) / window_size
    reminder = number_of_occurrence % window_size

    if len(extracted_input_text) <= window_size:
        window_input_text.append(extracted_input_text)
        window_target_text.append(extracted_target_text)
    else:
        counter1 = 1
        for selected_window in range(0, (int(division_result) * window_size), window_size):
            window_input_text.append(list(extracted_input_text[selected_window: counter1 * window_size]))

            j = list(extracted_target_text[selected_window: counter1 * window_size])
            j.append('\n')
            j = ['\t'] + j
            window_target_text.append(j)
            counter1 += 1
        if reminder != 0:
            number_of_missing_item = window_size - reminder
            # reminder_input_text = list(extracted_input_text[(len(extracted_input_text) - reminder): len(extracted_input_text)])
            # reminder_target_text = list(extracted_target_text[(len(extracted_target_text) - reminder): len(extracted_target_text)])

            window_input_text.append(
                list(extracted_input_text[(len(extracted_input_text) - reminder): len(extracted_input_text)]))

            window_target_text.append(
                list(extracted_target_text[(len(extracted_target_text) - reminder): len(extracted_target_text)]))

end_time = datetime.datetime.now()
print("takes : ", end_time - start_time)


for char in input_texts:
    if char not in input_characters:
        input_characters.add(char)
for char in target_texts:
    if char not in target_characters:
        target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters.add('\n')
target_characters.add('\t')
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = window_size
max_decoder_seq_length = window_size

print('Number of samples:', len(window_input_text))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(window_input_text), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(window_input_text), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(window_input_text), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


counter = 0

for (each_input_sent, each_target) in zip(window_input_text, window_target_text):

    print("we are processing sentence number:", counter)
    for t, (input_char, target_char) in enumerate(zip(each_input_sent, each_target)):
        try:
            encoder_input_data[counter, t, input_token_index[input_char]] = 1.
            decoder_input_data[counter, t, target_token_index[target_char]] = 1.
            if t > 0:
                decoder_target_data[counter, t - 1, target_token_index[target_char]] = 1.
        except:
            print(counter)

    counter += 1

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
