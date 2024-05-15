import yfinance as yf

# Define ticker symbol
ticker_symbol = 'AAPL'

# Define date range
start_date = '2019-01-01'
end_date = '2022-12-31'

# Define grid trading parameters
buy_interval = 0.05 # % below market price
sell_interval = 0.05 # % above market price
max_trades = 10 # maximum number of trades to execute at any given time

# Define transaction log
transactions = []

# Fetch data from yahoo finance API
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Get the latest closing price
latest_price = data.iloc[-1]['Close']

# Determine the buy/sell prices
buy_price = latest_price - (latest_price * buy_interval)
sell_price = latest_price + (latest_price * sell_interval)

# Check if we have sufficient funds to buy
if len(transactions) < max_trades:
    if buy_price < available_funds:
        # Place buy order
        transactions.append(('BUY', ticker_symbol, buy_price, 1, 0))
        available_funds -= buy_price

        # Place sell order
        transactions.append(('SELL', ticker_symbol, sell_price, 1, 0))

# Log transactions
print(transactions)
