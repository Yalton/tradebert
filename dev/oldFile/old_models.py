import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LinearRegressor(): 

    def __init__(self):
        pass

    def train(self, data): 
        # Read your data in and split the dependent and independent
        data = pd.read_csv('IBM.csv')
        X = data.drop(['Delta Close'], axis=1)
        y = data['Delta Close']

        # Train test spit
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Create the sequential
        model= Sequential()

        # Create the structure of the neural network
        model.add(Dense(1, input_shape=(X_train.shape[1],), activation='tanh'))
        model.add(Dense(3, activation='tanh'))
        model.add(Dense(3, activation='tanh'))
        model.add(Dense(3, activation='tanh'))
        model.add(Dense(1, activation='tanh'))

        # Compile the model
        model.compile(
                      optimizer='rmsprop',
                      loss='hinge',
                      metrics=['accuracy']
                      )
        # Train the model
        model.fit(X_train.values, y_train.values, epochs=100)

        # Evaluate the predictions of the model
        y_pred = model.predict(X_test.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y_test, y_pred))

        # Save structure to json
        model = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model)

        # Save weights to HDF5
        model.save_weights("LinearRegressor.h5")

    def predict(self):
        pass

class RNN(): 

    def __init__(self): 
        pass

    def train(self, data): 
        # load data
        prices = data['Close'].values.reshape(-1, 1)

        # normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = scaler.fit_transform(prices)

        # prepare data
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        look_back = 10
        X, Y = create_dataset(prices, look_back)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # create and fit the RNN network
        model = Sequential()
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=100, batch_size=32)

        # make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # invert predictions
        train_predict = scaler.inverse_transform(train_predict)
        Y_train = scaler.inverse_transform([Y_train])
        test_predict = scaler.inverse_transform(test_predict)
        Y_test = scaler.inverse_transform([Y_test])


    def predict(self, data): 
        pass
    
class LTSM(): 
    def __init__(self): 
        pass

    def train(self, data):
        prices = data['Close'].values.reshape(-1, 1)

        # normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = scaler.fit_transform(prices)

        # prepare data
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        look_back = 10
        X, Y = create_dataset(prices, look_back)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=100, batch_size=32)

        # make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # invert predictions
        train_predict = scaler.inverse_transform(train_predict)
        Y_train = scaler.inverse_transform([Y_train])
        test_predict = scaler.inverse_transform(test_predict)
        Y_test = scaler.inverse_transform([Y_test])

    def predict(self, data):
        pass