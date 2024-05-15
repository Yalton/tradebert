import yfinance as yf
import pandas as pd

# Retrieve historical data for Alibaba
alibaba = yf.Ticker("BABA")
df = alibaba.history(period="max")

# Compute Simple Moving Average
sma_20 = df.Close.rolling(window=20).mean()

# Predict future prices using Simple Moving Average
df['SMA_20'] = sma_20
df['Prediction'] = df.SMA_20.shift(-1)

# Drop NaN rows from DataFrame
df.dropna(inplace=True)

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Define input and output data for training and testing
X_train = train_data[['Close', 'SMA_20']]
X_test = test_data[['Close', 'SMA_20']]
y_train = train_data['Prediction']
y_test = test_data['Prediction']

# Train model and print accuracy score
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
