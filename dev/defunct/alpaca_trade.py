import alpaca_trade_api as tradeapi

api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
base_url = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

# Initialize Alpaca API client
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Function to perform market buy
def market_buy(symbol, qty):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    return order

# Function to perform market sell
def market_sell(symbol, qty):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
    return order

# Function to perform limit buy
def limit_buy(symbol, qty, price):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='limit',
        time_in_force='gtc',
        limit_price=price
    )
    return order

# Function to perform limit sell
def limit_sell(symbol, qty, price):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='limit',
        time_in_force='gtc',
        limit_price=price
    )
    return order

# Example usage
symbol = 'AAPL'
qty = 1
buy_price = 120
sell_price = 130

# Market buy
market_buy_order = market_buy(symbol, qty)
print("Market Buy Order:", market_buy_order)

# Market sell
market_sell_order = market_sell(symbol, qty)
print("Market Sell Order:", market_sell_order)

# Limit buy
limit_buy_order = limit_buy(symbol, qty, buy_price)
print("Limit Buy Order:", limit_buy_order)

# Limit sell
limit_sell_order = limit_sell(symbol, qty, sell_price)
print("Limit Sell Order:", limit_sell_order)