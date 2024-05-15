import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# specify the list of cryptocurrencies and stocks you want to download
symbol_list = ['BTC', 'ETH', 'AMZN', 'AAPL', 'FB', 'GOOG', 'MSFT', 'NFLX', 'TSLA']

# define the directory to save the downloaded data
dir_path = './data/'

# create the directory if it does not exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# define the range of dates to download weekly data
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=365)
delta = timedelta(weeks=1)

# function to download data from cryptocompare
def download_cryptocompare_data(symbol):
    url = 'https://min-api.cryptocompare.com/data/histoday'
    params = {'fsym': symbol, 'tsym': 'USD', 'limit': '2000'}

    # download data from cryptocompare
    response = requests.get(url, params=params)
    data = response.json()['Data']

    # create a pandas dataframe from the downloaded data
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.date

    # filter the data based on the specified date range
    mask = (df['time'] >= start_date) & (df['time'] <= end_date)
    df = df.loc[mask]

    # save the data to a CSV file
    file_path = os.path.join(dir_path, f'{symbol}_cryptocompare.csv')
    df.to_csv(file_path, index=False)

# function to download data from yahoo finance
def download_yahoo_finance_data(symbol):
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}'
    params = {'interval': '1wk', 'range': '1y', 'events': 'history', 'includeAdjustedClose': 'true'}

    # download data from yahoo finance
    response = requests.get(url, params=params)
    data = response.content.decode().split('\n')
    headers = data[0].split(',')
    headers[0] = 'date'

    # create a pandas dataframe from the downloaded data
    df = pd.DataFrame(columns=headers)
    for row in data[1:-1]:
        row_data = row.split(',')
        row_data[0] = datetime.strptime(row_data[0], "%Y-%m-%d").date()
        df = df.append(pd.Series(row_data, index=headers), ignore_index=True)

    # filter the data based on the specified date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask]

    # save the data to a CSV file
    file_path = os.path.join(dir_path, f'{symbol}_yahoo_finance.csv')
    df.to_csv(file_path, index=False)

# loop through the list of symbols and download the data
for symbol in symbol_list:
    print(f'Downloading data for {symbol}')
    download_cryptocompare_data(symbol)
    download_yahoo_finance_data(symbol)
    print('Data download completed successfully!')