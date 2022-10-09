import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')


def getData(month_count):
    # import dataset
    df = pd.read_csv('new_data1.csv', sep=",", index_col='Date', parse_dates=['Date'])
    # dealing with monthly data
    df.index.freq = 'MS'

    df.head()

    results = seasonal_decompose(df['Bill'])

    # training data part
    train = df.iloc[:174]
    # test = df.iloc[160:]
    # convert to scale of 0 to 1
    scaler = MinMaxScaler()

    # fit the scalar to train set
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    # scaled_test = scaler.transform(test)

    # give input and output to supervised learning
    n_features = 1
    n_input = 12
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    # model.fit(generator, epochs=25)

    # loss_per_epoch = model.history.history['loss']
    # plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

    # get the prediction
    last_train_batch = scaled_train[-12:]
    last_train_batch = last_train_batch.reshape((1, n_input, n_features))
    # model.save('model.h5')

    model.predict(last_train_batch)

    # predictions with training set
    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    # print("length is " + str(len(test)))

    value = int(month_count)

    for i in range(value):
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]

        # append the prediction into the array
        test_predictions.append(current_pred)

        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # test.head()
    # transform back to original
    true_predictions = scaler.inverse_transform(test_predictions)

    tt = true_predictions.reshape(-1)

    data = []
    i = 0
    for x in tt:
        obj = {'index': i, 'value': x}
        i = i + 1
        data.append(obj)

    return data

# print(getData(6))
