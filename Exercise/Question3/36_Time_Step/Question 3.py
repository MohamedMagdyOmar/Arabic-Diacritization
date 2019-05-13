import numpy as np
import pandas as pd
from keras.layers import Dropout, LSTM, Dense, Input, SpatialDropout1D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os


def feature_scaling(pd_data_frame):
    mean = pd_data_frame['IPG3113N'].mean()
    max_value = pd_data_frame['IPG3113N'].max()
    min_value = pd_data_frame['IPG3113N'].min()
    std_deviation = max_value - min_value
    norm = (pd_data_frame['IPG3113N'] - mean) / std_deviation
    return norm


def preprocess_dataset(df):
    m = len(df) - NUM_TIME_STEPS - 1
    X = np.zeros((m, NUM_TIME_STEPS, 12))
    Y = np.zeros(m)

    for i in range(m):
        for j in range(NUM_TIME_STEPS):
            X[i, j, j % 12] = df.iloc[i + j]

        # now we have filled first row with 36 months, and now we are predicting month number 37
        Y[i] = df.iloc[i + NUM_TIME_STEPS]

    # default test size = 0.25
    return train_test_split(X, Y, random_state=1)


if __name__ == "__main__":

    print(os.listdir("../36_Time_Step"))

    NUM_TIME_STEPS = 36

    data_frame = pd.read_csv("../36_Time_Step/candy_production.csv")
    data_frame_norm = feature_scaling(data_frame)

    X_train, X_test, Y_train, Y_test = preprocess_dataset(data_frame_norm)

    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)
    print("X_test", X_test.shape)
    print("Y_test", Y_test.shape)

    X_input = Input((NUM_TIME_STEPS, 12), dtype="float32")

    X = SpatialDropout1D(0.3)(X_input)

    X = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)

    X = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(X)

    # Propagate X through a Dense layer without activation to get back a batch of 1-dimensional vectors.
    X = Dense(1)(X)

    model = Model(inputs=X_input, outputs=X)

    model.summary()

    model.compile(loss='mse', optimizer='Adam')

    monitor = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True)

    hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, callbacks=[monitor, checkpoint],
                     validation_split=0.01, shuffle=True)

    model.load_weights('best_weights.hdf5')
    model.save('final_model.h5')

    '''
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, name="LSTM Layer 1"))
    model.add(Dropout(0.2))

    model.add(LSTM(128, name="LSTM Layer 2"))
    model.add(Dropout(0.2))

    model.add(Dense(Y_train.shape[0], activation='softmax', name="dense"))

    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

    model.fit(X_train, Y_train, callbacks=[early_stop, checkpoint], epochs=30, batch_size=64, verbose=1,
              validation_split=0.01, shuffle=True)

    model.load_weights('best_weights.hdf5')
    model.save('final_model.h5')
    '''