from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import data_helper as dp


def createModel(nClasses, input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='valid', strides=1, activation='relu', input_shape=(13, 3, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model

dp.establish_db_connection()

data = dp.load_dataset_table("training")
train_data = dp.load_nn_input_dataset(data[:, [0, 8]])
train_labels_one_hot = dp.load_nn_labels_dataset(data[:, [0, 1]])

data = dp.load_dataset_table("testing")
test_data = dp.load_nn_input_dataset(data[:, [0, 8]])
test_labels_one_hot = dp.load_nn_labels_dataset(data[:, [0, 1]])

model1 = createModel(49, (39,))
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(test_data, test_labels_one_hot))

model1.evaluate(test_data, test_labels_one_hot)