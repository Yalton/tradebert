import ccxt

api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'

# Initialize Binance client
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret
})

# Function to perform market buy
def market_buy(symbol, amount):
    order = binance.create_market_buy_order(symbol, amount)
    return order

# Function to perform market sell
def market_sell(symbol, amount):
    order = binance.create_market_sell_order(symbol, amount)
    return order

# Function to perform limit buy
def limit_buy(symbol, amount, price):
    order = binance.create_limit_buy_order(symbol, amount, price)
    return order

# Function to perform limit sell
def limit_sell(symbol, amount, price):
    order = binance.create_limit_sell_order(symbol, amount, price)
    return order

# Example usage
symbol = 'BTC/USDT'
amount = 0.001
buy_price = 45000
sell_price = 50000

# Market buy
market_buy_order = market_buy(symbol, amount)
print("Market Buy Order:", market_buy_order)

# Market sell
market_sell_order = market_sell(symbol, amount)
print("Market Sell Order:", market_sell_order)

# Limit buy
limit_buy_order = limit_buy(symbol, amount, buy_price)
print("Limit Buy Order:", limit_buy_order)

# Limit sell
limit_sell_order = limit_sell(symbol, amount, sell_price)
print("Limit Sell Order:", limit_sell_order)