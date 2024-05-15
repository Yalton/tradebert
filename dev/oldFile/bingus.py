import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import ccxt

# Set Yahoo Finance API credentials
yf.pdr_override()
yahoo_api = 'YOUR_YAHOO_API_KEY'

# Create a list of stock and cryptocurrency tickers
stocks_list = ['AAPL', 'GOOGL', 'TSLA', 'IBM']
crypto_list = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT']

# Initialize ccxt exchange object
binance = ccxt.binance()

# Create empty dataframe to store the historical data
historical_data = pd.DataFrame()

# Load historical data for stocks
for ticker in stocks_list:
    data = pdr.get_data_yahoo(ticker, start='2020-01-01', end='2021-06-30')
    data['Ticker'] = ticker
    historical_data = historical_data.append(data)

# Load historical data for cryptocurrencies
for symbol in crypto_list:
    data = pd.DataFrame(binance.fetch_ohlcv(symbol, '1d', limit=1000), columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data.set_index('Timestamp', inplace=True)
    data['Ticker'] = symbol
    historical_data = historical_data.append(data)

# Print the historical data
print(historical_data.head())