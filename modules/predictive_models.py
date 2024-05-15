import numpy as np
import pandas as pd
import os 
import joblib
from math import sqrt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.utils import check_random_state

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam


def preProcess(data):
    # Check for NaN values
    print(data.isnull().sum())

    # Drop rows with NaN values
    data = data.dropna()

    data = data.select_dtypes(exclude=['object'])

    return data

# Abstract Base Class
# class PredictiveModels: 
#     def __init__(self, symbol): 
#         self.model = Sequential()
#         self.symbol = symbol


class LinearRegressor(): 

    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol
        if not os.path.exists(f"models/{self.symbol}/"):
            os.makedirs(f"models/{self.symbol}/")

    def train(self, data): 

        # preprocess the data
        #data, _ = preProcess(data)
        data = preProcess(data)

        X = data.drop(['Close', 'Adj Close'], axis=1)  # drop 'Adj Close'
        y = data['Close']

        # Fit the scaler to the 'Close' column
        self.y_scaler = MinMaxScaler().fit(y.values.reshape(-1, 1))

        # Transform the 'Close' column
        y = self.y_scaler.transform(y.values.reshape(-1, 1))

        # Fit the scaler to the X data
        self.x_scaler = MinMaxScaler().fit(X)

        # Transform the X data
        X = self.x_scaler.transform(X)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # set a random seed for reproducibility

        self.model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # larger layer with 'relu' activation
        self.model.add(Dense(64, activation='relu'))  # 'relu' activation
        self.model.add(Dense(32, activation='relu'))  # 'relu' activation
        self.model.add(Dense(1))  # linear activation by default

        self.model.compile(optimizer='adam', loss='mean_squared_error')  # 'adam' optimizer

        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=2)  # add validation data and change verbosity

        y_pred = self.model.predict(X_test)


        #y_pred = self.model.predict(X_test)

        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred)
        Y_test_unscaled = self.y_scaler.inverse_transform(Y_test)  # Unscale the true target values

        mse = mean_squared_error(Y_test_unscaled, y_pred_unscaled)
        mae = mean_absolute_error(Y_test_unscaled, y_pred_unscaled)
        self.r2 = r2_score(Y_test_unscaled, y_pred_unscaled)

        print('Mean Squared Error:', mse)
        print('Mean Absolute Error:', mae)
        print('R^2 Score:', self.r2)

        print("Inverse Transformed y_pred", y_pred_unscaled)

        self.model.save(f"models/{self.symbol}/LinearRegressor.h5")

        

    def predict(self, X):
        self.model = load_model(f"models/{self.symbol}/LinearRegressor.h5")

        # Ensure that the input X also drops the 'Close' and 'Adj Close' columns
        if 'Close' in X.columns:
            X = X.drop(['Close', 'Adj Close'], axis=1)

        X = X.select_dtypes(exclude=['object'])
        #X = X.values.astype('float32')

        # Scale the X data with the scaler used in training
        X = self.x_scaler.transform(X)

        predictions = self.model.predict(X)

        predictions = self.y_scaler.inverse_transform(predictions)

        return predictions
        
    


class RNN(): 
    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol
        if not os.path.exists(f"models/{self.symbol}/"):
            os.makedirs(f"models/{self.symbol}/")

    def train(self, data): 

        data = preProcess(data)

        X = data.drop(['Close', 'Adj Close'], axis=1)  
        y = data['Close']

        # Fit the scaler to the 'Close' column
        self.y_scaler = MinMaxScaler().fit(y.values.reshape(-1, 1))

        # Transform the 'Close' column
        y = self.y_scaler.transform(y.values.reshape(-1, 1))

        # Fit the scaler to the X data
        self.x_scaler = MinMaxScaler().fit(X)

        # Transform the X data
        X = self.x_scaler.transform(X)

        look_back = 10
        
        X, Y = self.create_dataset(X, look_back)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        self.model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(SimpleRNN(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        self.model.fit(X_train, Y_train, epochs=100, batch_size=32)

        y_pred = self.model.predict(X_test)

        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred)
        Y_test_unscaled = self.y_scaler.inverse_transform(Y_test.reshape(-1, 1))  # Unscale the true target values

        mse = mean_squared_error(Y_test_unscaled, y_pred_unscaled)
        mae = mean_absolute_error(Y_test_unscaled, y_pred_unscaled)
        self.r2 = r2_score(Y_test_unscaled, y_pred_unscaled)

        print('Mean Squared Error:', mse)
        print('Mean Absolute Error:', mae)
        print('R^2 Score:', self.r2)

        self.model.save(f"models/{self.symbol}/RNN.h5")

        


    def predict(self, X): 
        self.model = load_model(f"models/{self.symbol}/RNN.h5")

        # Ensure that the input X also drops the 'Close' and 'Adj Close' columns
        if 'Close' in X.columns:
            X = X.drop(['Close', 'Adj Close'], axis=1)

        X = X.select_dtypes(exclude=['object'])

        # Scale the X data with the scaler used in training
        X = self.x_scaler.transform(X)

        # Reshape the data to include 10 time steps. If there are not enough previous samples, pad with zeros
        X = np.array([np.concatenate([np.zeros((10 - len(X[i:]), X.shape[1])), X[i: i+10]]) if len(X[i: i+10]) < 10 else X[i: i+10] for i in range(len(X))])

        predictions = self.model.predict(X)

        predictions = self.y_scaler.inverse_transform(predictions)

        return predictions

    
    @staticmethod
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :]  # <-- Note the change here from '0' to ':'
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)




class LSTMModel(): 
    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol
        if not os.path.exists(f"models/{self.symbol}/"):
            os.makedirs(f"models/{self.symbol}/")

    def train(self, data):

        data = preProcess(data)

        X = data.drop(['Close', 'Adj Close'], axis=1)  
        y = data['Close']

        # Fit the scaler to the 'Close' column
        self.y_scaler = MinMaxScaler().fit(y.values.reshape(-1, 1))

        # Transform the 'Close' column
        y = self.y_scaler.transform(y.values.reshape(-1, 1))

        # Fit the scaler to the X data
        self.x_scaler = MinMaxScaler().fit(X)

        # Transform the X data
        X = self.x_scaler.transform(X)

        look_back = 10
        
        X, Y = self.create_dataset(X, look_back)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        self.model.fit(X_train, Y_train, epochs=100, batch_size=32)

        y_pred = self.model.predict(X_test)

        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred)
        Y_test_unscaled = self.y_scaler.inverse_transform(Y_test.reshape(-1, 1))  # Unscale the true target values

        mse = mean_squared_error(Y_test_unscaled, y_pred_unscaled)
        mae = mean_absolute_error(Y_test_unscaled, y_pred_unscaled)
        self.r2 = r2_score(Y_test_unscaled, y_pred_unscaled)

        print('Mean Squared Error:', mse)
        print('Mean Absolute Error:', mae)
        print('R^2 Score:', self.r2)

        self.model.save(f"models/{self.symbol}/LSTM.h5")

        

    def predict(self, X): 
        self.model = load_model('models/LSTM.h5')
        # Ensure that the input X also drops the 'Close' and 'Adj Close' columns
        if 'Close' in X.columns:
            X = X.drop(['Close', 'Adj Close'], axis=1)

        X = X.select_dtypes(exclude=['object'])

        # Scale the X data with the scaler used in training
        X = self.x_scaler.transform(X)

        # Reshape the data to include 10 time steps. If there are not enough previous samples, pad with zeros
        X = np.array([np.concatenate([np.zeros((10 - len(X[i:]), X.shape[1])), X[i: i+10]]) if len(X[i: i+10]) < 10 else X[i: i+10] for i in range(len(X))])

        predictions = self.model.predict(X)

        predictions = self.y_scaler.inverse_transform(predictions)

        return predictions
        
    @staticmethod
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :] 
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)


class HMM: 
    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol
        if not os.path.exists(f"models/{self.symbol}/"):
            os.makedirs(f"models/{self.symbol}/")

    def train(self, data): 
        data = preProcess(data)

        # Extract the Close column
        self.close_data = data[["Close"]].values

        # Fit an HMM model
        self.model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
        self.model.fit(self.close_data)

        # Save the model
        joblib.dump(self.model, f"models/{self.symbol}/HMM.pkl")

    def hiddenStates(self): 
        # Load the model
        self.model = joblib.load(f"models/{self.symbol}/HMM.pkl")

        # Predict hidden states
        hidden_states = self.model.predict(self.close_data)

        return hidden_states

    def predict(self, days): 
        # Load the model
        self.model = joblib.load(f"models/{self.symbol}/HMM.pkl")

        # Start with the last day in the training set
        last_day = self.close_data[-1]
        
        # Predict the hidden states for the next few days
        random_state = check_random_state(None)
        next_state = (self.model.transmat_.cumsum(axis=1) > random_state.rand(self.model.n_components)).argmax(axis=1)
        
        # Predict the close prices for the next few days
        next_obs = np.zeros(days)
        for i in range(days):
            next_obs[i] = np.dot(self.model.means_.ravel(), self.model.predict_proba([last_day])[0])
            next_state = (self.model.transmat_[next_state].cumsum() > random_state.rand()).argmax()
        last_day = next_obs[i]
    
        return next_obs
    
    def print_details(self): 
        if self.model is None: 
            print("Model not trained yet")
            return

        # Print state transitions
        print("Transition matrix")
        print(self.model.transmat_)

        # Print the means and covariances of each state
        for i in range(self.model.n_components):
            print(f"Mean of state {i+1}: {self.model.means_[i][0]:.2f}")
            print(f"Covariance of state {i+1}:")
            print(self.model.covars_[i])
            print()

class ARIMAModel():
    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol

    def train(self, data):
        data = preProcess(data)

        X = data.drop(['Close', 'Adj Close'], axis=1)  
        y = data['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = ARIMA(y_train, order=self.order)
        self.model_fit = self.model.fit()

        # make predictions
        y_pred = self.model_fit.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, dynamic=False)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        print('RMSE: %.3f' % rmse)

    def predict(self, X): 
        # Ensure that the input X also drops the 'Close' and 'Adj Close' columns
        if 'Close' in X.columns:
            X = X.drop(['Close', 'Adj Close'], axis=1)

        X = X.select_dtypes(exclude=['object'])
        
        predictions = self.model_fit.predict(start=1, end=len(X), dynamic=False)
        
        return predictions

    def forecast(self, days): 
        forecast = self.model_fit.forecast(steps=days)
        return forecast

class GARCH():

    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol

    def train(self, data):
        #data = preProcess(data)


        returns = data['Close'].pct_change().dropna()
        model = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = model.fit()

    def predict(self, X): 
        # Ensure that the input X also drops the 'Close' and 'Adj Close' columns
        if 'Close' in X.columns:
            X = X.drop(['Close', 'Adj Close'], axis=1)

        X = X.select_dtypes(exclude=['object'])
        
        predictions = self.model_fit.predict(start=1, end=len(X), dynamic=False)
        
        return predictions

    def forecast(self, days): 
        forecast = self.model_fit.forecast(steps=days)
        return forecast

class GeometricBrownianMotion(): 
    def __init__(self, symbol): 
        self.model = Sequential()
        self.symbol = symbol
        
    def predict_prices(self, data, days):
        data['LogReturn'] = np.log(data['Close']).shift(-1) - np.log(data['Close'])

        # Calculate drift and volatility
        u = data['LogReturn'].mean() # Mean of the logarithmic return
        var = data['LogReturn'].var() # Variance of the logarithmic return
        drift = u - (0.5 * var) # Drift coefficient
        stdev = data['LogReturn'].std() # Standard deviation of the logarithmic return

        # Perform simulation
        price = data['Close'].iloc[-1] # Start price
        prices = [price]

        for _ in range(days):
            shock = norm.ppf(np.random.rand()) * stdev
            price = price * np.exp(drift + shock)
            prices.append(price)
            
        return prices


# returns = data['Close'].pct_change().dropna()
# model = arch_model(returns, vol='Garch', p=1, q=1)
# model_fit = model.fit()


# Evaluate the model
# test_loss, test_mae = self.model.evaluate(X_test, Y_test)

# print('Test loss:', test_loss)
# print('Test MAE:', test_mae)


# self.model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# self.model.add(Dropout(0.2))
# self.model.add(SimpleRNN(units=50))
# self.model.add(Dropout(0.2))
# self.model.add(Dense(1))

# self.model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# self.model.add(Dropout(0.2))
# self.model.add(SimpleRNN(100, return_sequences=True))
# self.model.add(Dropout(0.2))
# self.model.add(SimpleRNN(100))
# self.model.add(Dropout(0.2))
# self.model.add(Dense(1))