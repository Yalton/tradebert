import yfinance as yf

# Define ticker symbol
ticker_symbol = 'AAPL'

# Define date range
start_date = '2019-01-01'
end_date = '2022-12-31'

# Fetch data from yahoo finance API
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# View the data
print(data)


# Similarly, we can obtain historical data on cryptocurrencies by passing the crypto ticker symbol to the `yf.Ticker()` function as shown:


import yfinance as yf

# Define crypto ticker symbol
crypto_symbol = 'BTC-USD'

# Define date range
start_date = '2019-01-01'
end_date = '2022-12-31'

# Fetch data from yahoo finance API for Cryptocurrency
crypto_data = yf.download(crypto_symbol, start=start_date, end=end_date)

# View the data
print(crypto_data)
