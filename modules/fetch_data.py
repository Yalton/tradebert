import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
import yfinance as yf


class DataAggregator:
    def __init__(self): 
        pass

    def _make_request(self, url, params=None):
        """A helper method to make HTTP requests and handle exceptions."""
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Something went wrong... : {err}")
        return None

        # function to download data from cryptocompare
        def download_cryptocompare_data(symbol, toCSV):
            url = 'https://min-api.cryptocompare.com/data/histoday'
            params = {'fsym': symbol, 'tsym': 'USD', 'limit': '2000'}

            # download data from cryptocompare
            response = requests.get(url, params=params)

            # data = response.json()['Data']

            data = response.json()['Data']

            # create a pandas dataframe from the downloaded data
            df = pd.DataFrame(data)

            df.to_csv("csvs/debug.csv")

            df['time'] = pd.to_datetime(df['time'], unit='s').dt.date

            # filter the data based on the specified date range
            mask = (df['time'] >= start_date) & (df['time'] <= end_date)
            df = df.loc[mask]

            # save the data to a CSV file
            if(toCSV):
                file_path = os.path.join(dir_path, f'{symbol}_cryptocompare.csv')
                df.to_csv(file_path, index=False)
            return df 
            
        # loop through the list of symbols and download the data
        datalist = []
        for symbol in symbol_list:
            print(f'Downloading data for {symbol}')
            datalist.append(download_cryptocompare_data(symbol, toCSV))
            #download_yahoo_finance_data(symbol)
            print('Data download completed successfully!')

    # def fetch_yfinance_api(self, symbol, start_date, end_date):
    #     """Fetch data from Yahoo Finance API for a specific symbol."""
    #     try:
    #         data = yf.download(symbol, start=start_date, end=end_date)
    #         #print(data)
    #         data = data.rename(columns={'Adj Close': 'Adj_Close'})
    #         return data
    #     except Exception as e:
    #         print(f"An error occurred while fetching data from Yahoo Finance API: {e}")

    def fetch_yfinance_api(self, symbol, start_date, end_date, interval='15m'):
        """Fetch data from Yahoo Finance API for a specific symbol."""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            data = data.rename(columns={'Adj Close': 'Adj_Close'})
            # print("Yfinance data is", data)
            return data
        except Exception as e:
            print(f"An error occurred while fetching data from Yahoo Finance API: {e}")



    def whale_watcher(self): 
        pass

# # function to download data from yahoo finance
# def download_yahoo_finance_data(symbol):
#     url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}'
#     params = {'interval': '1wk', 'range': '1y', 'events': 'history', 'includeAdjustedClose': 'true'}

#     # download data from yahoo finance
#     response = requests.get(url, params=params)
#     data = response.content.decode().split('\n')
#     headers = data[0].split(',')
#     headers[0] = 'date'

#     # create a pandas dataframe from the downloaded data
#     df = pd.DataFrame(columns=headers)
#     for row in data[1:-1]:
#         row_data = row.split(',')
#         row_data[0] = datetime.strptime(row_data[0], "%Y-%m-%d").date()
#         df = df.append(pd.Series(row_data, index=headers), ignore_index=True)

#     # filter the data based on the specified date range
#     mask = (df['date'] >= start_date) & (df['date'] <= end_date)
#     df = df.loc[mask]

#     # save the data to a CSV file
#     file_path = os.path.join(dir_path, f'{symbol}_yahoo_finance.csv')
#     df.to_csv(file_path, index=False)



    # def extract_financial_news(self):
    #     """Extract financial news articles from MarketWatch."""
    #     url = 'https://www.marketwatch.com/newsviewer'
    #     response = self._make_request(url)
    #     if response is not None:
    #         content = response.content
    #         soup = BeautifulSoup(content, 'html.parser')
    #         articles = soup.find_all('div', {'class': 'article__content'})
    #         return [article for article in articles if 'marketwatch.com/story/' in article.find('a')['href']]

    # def fetch_asset_prices(self, symbol_list, dir_path, toCSV):
    #     """Fetch asset prices from CryptoCompare and Yahoo Finance."""
    #      # # specify the list of cryptocurrencies and stocks you want to download
    #     # symbol_list = ['BTC', 'ETH', 'AMZN', 'AAPL', 'FB', 'GOOG', 'MSFT', 'NFLX', 'TSLA']

    #     # # define the directory to save the downloaded data
    #     # dir_path = './data/'

    #     # create the directory if it does not exist
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    #     # define the range of dates to download weekly data
    #     end_date = datetime.utcnow().date()
    #     start_date = end_date - timedelta(days=365)
    #     delta = timedelta(weeks=1)

    # def fetch_crypto_news(self):
    #     """Fetch top 10 cryptocurrency news from Google News."""
    #     search_terms = "current cryptocurrency market trends"
    #     url = f"https://www.google.com/search?q={search_terms}&tbm=nws"
    #     res = self._make_request(url)
    #     if res is not None:
    #         soup = BeautifulSoup(res.text, "html.parser")
    #         search_results = soup.select(".dbsr")
    #         top_ten_results = [result.a.text for result in search_results[:10]]
    #         for i, result in enumerate(top_ten_results, start=1):
    #             print(f"{i}. {result}")
    #     else: 
    #         print("Received no request from the fetch")