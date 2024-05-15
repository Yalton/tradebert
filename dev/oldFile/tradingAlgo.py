# Required `pandas`, `numpy`, `yfinance`, `tensorflow`, `keras`, `sklearn`, `datetime`, `talib`, `mplfinance`, and `matplotlib`.

import pandas as pd 
import numpy as np 
import yfinance as yf 
import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from sklearn.preprocessing import MinMaxScaler 
from datetime import datetime 
import talib 
import mplfinance as mpf 
import matplotlib.pyplot as plt

# Bollinger Band (BB) trading strategy:
def bollinger_bands_strategy(ohlcv_data):
    upper, middle, lower = talib.BBANDS(np.array(ohlcv_data['Close']), timeperiod=20)
    ohlcv_data['BBUP'] = upper
    ohlcv_data['BBLOW'] = lower

    bb_signal = ""
    if ohlcv_data['Close'].iloc[-1] > upper[-1]:
        bb_signal = "SELL"
    elif ohlcv_data['Close'].iloc[-1] < lower[-1]:
        bb_signal = "BUY"
    else:
        bb_signal = "HOLD"
        
    return bb_signal

# Moving Average Crossover (MAC) trading strategy 
def mac_strategy(ohlcv_data):
    short_window = 50
    long_window = 200

    signals = pd.DataFrame(index=ohlcv_data.index)
    signals['signal'] = 0.0

    signals['short_mavg'] = ohlcv_data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = ohlcv_data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    if signals['positions'].iloc[-1] == 1.0:
        mac_signal = "BUY"
    elif signals['positions'].iloc[-1] == -1.0:
        mac_signal = "SELL"
    else:
        mac_signal = "HOLD"
        
    return mac_signal

# Long Short Term Memory (LSTM) model for stock price prediction:
def lstm_prediction(pred_date, symbol):
    st = datetime(2000, 1, 1)
    en = datetime.now()

    df = yf.download(symbol, start=st, end=en)

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))

    training_size=int(len(prices_scaled)*0.7)

    X_train = []
    y_train = []
    for i in range(60, training_size):
        X_train.append(prices_scaled[i-60:i,0])
        y_train.append(prices_scaled[i,0])

    X_test = []
    y_test = []
    for i in range(training_size, len(prices_scaled)):
        X_test.append(prices_scaled[i-60:i,0])
        y_test.append(prices_scaled[i,0])


    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Build the model 
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history=model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test,y_test), verbose=1, shuffle=False)


    test_st = pred_date
    test_en = datetime.now()

    dfs = yf.download(symbol, start=test_st, end=test_en)

    test_set = dfs['Adj Close'].values.reshape(-1,1)
    test_set_scaled = scaler.fit_transform(test_set)

    X_test = []
    for i in range(60, len(test_set_scaled)):
        X_test.append(test_set_scaled[i-60:i,0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    closing_price = model.predict(X_test)

    closing_price = scaler.inverse_transform(closing_price)


    return closing_price[-1][0]

# Function to plot.
def plot_data(ohlcv_data):
    ohlcv_data = ohlcv_data[-90:]
    ohlcv_data['12-SMA'] = ohlcv_data.Close.rolling(window=12).mean()
    ohlcv_data['26-SMA'] = ohlcv_data.Close.rolling(window=26).mean()
    mpf.plot(ohlcv_data,type='line',mav=(12,26),figsize=(15,7), title='Technical Analysis',
             ylabel='Price (USD)')

# Execute the trading strategies:
def execute_strategies(symbol):
    symbol_data = yf.Ticker(symbol).history(period="max")
    bb_signal = bollinger_bands_strategy(symbol_data)
    mac_signal = mac_strategy(symbol_data)
    lstm_pred = lstm_prediction(datetime(2023, 4, 15), symbol)
    
    if bb_signal == "BUY" and mac_signal == "BUY":
        signal = "STRONG BUY"
    elif bb_signal == "BUY" and mac_signal == "HOLD":
        signal =  "BUY"
    elif bb_signal == "HOLD" and mac_signal == "BUY":
        signal = "BUY"
    elif bb_signal == "SELL" and mac_signal == "SELL":
        signal = "STRONG SELL"
    elif bb_signal == "SELL" and mac_signal == "HOLD":
        signal = "SELL"
    elif bb_signal == "HOLD" and mac_signal == "SELL":
        signal = "SELL"
    else: 
        signal = "HOLD"
    
    return {"symbol":symbol, "Bollinger Band signal":bb_signal, 
            "Moving Average Crossover signal":mac_signal, 
            "LSTM prediction for next day":lstm_pred, 
            "Signal":signal}


# Test the trading algorithm:
if __name__ == "__main__":
    symbol = "AAPL"
    print(execute_strategies(symbol))