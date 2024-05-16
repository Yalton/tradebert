import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=Y7DO72A98FXEONS9'
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=15min&outputsize=full&apikey=Y7DO72A98FXEONS9&datatype=json&start=2023-05-16&enddate=2024-05-15'

r = requests.get(url)
data = r.json()

print(data)