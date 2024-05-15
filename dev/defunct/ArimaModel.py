import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# retrieve historical market data for Tesla for the past 5 years
tsla_data = yf.Ticker("TSLA").history(period="5y")

# fit ARIMA model to the data
model = ARIMA(tsla_data['Close'], order=(1, 0, 0))
model_fit = model.fit()

# predict the next 100 days of Tesla closing prices
forecast = model_fit.forecast(steps=100)

# print the forecasted values
print(forecast)
