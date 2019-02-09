# this code comes from below website with some modification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy
import data_helper as dp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.callbacks import TensorBoard
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
import RepositoryLayer.Repository as Repository
import ServiceLayer.PreprocessingServices as PreprocessService
# fix random seed for reproducibility
numpy.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Arabic_CNN:

    def __init__(self):
        self.req_char_index = 13
        self.window_size = 17
        self.repo = Repository.Repository()
        self.service = PreprocessService.Services()
        self.x = ''
        self.y = ''

    def get_data_from_db(self, category):

        dataset = self.repo.get_dataset_by_category(category)

        self.x = dataset[:, [0, 3]]
        self.y = dataset[:, [1, 3]]

    def preprocess_dataset(self):

        padded_sentences = self.service.preprocess(self.x, self.req_char_index, self.window_size)

        vocabulary, vocabulary_inv = self.service.build_vocab(padded_sentences)

        padded_sentence = self.service.build_input_data(vocabulary)

        return padded_sentence, self.y, vocabulary, vocabulary_inv

    def check_dataset_keys_are_matched(self, vocab_training, vocab_testing):

        if any(key in vocab_training for key in vocab_testing):
            pass
        else:
            raise Exception("some keys in training data set not found in testing data set")


if __name__ == "__main__":
    Arabic_CNN = Arabic_CNN()

    X_train, y_train, vocabulary_train, vocabulary_inv_train = Arabic_CNN.preprocess_dataset('training')
    X_train = (numpy.arange(X_train.max()) == X_train[..., None] - 1).astype(int)

    X_test, y_test, vocabulary_test, vocabulary_inv_test = Arabic_CNN.preprocess_dataset('testing')
    X_test = (numpy.arange(X_test.max()) == X_test[..., None] - 1).astype(int)

    Arabic_CNN.check_key_exist(vocabulary_train, vocabulary_test)

    model = Sequential()

    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(window_size, 37)))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    callbacks_list = [checkpoint, early_stop]

    tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              callbacks=callbacks_list, epochs=40, batch_size=79, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))