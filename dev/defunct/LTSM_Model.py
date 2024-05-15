import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Define a function to extract historical data from Alpha Vantage
def extract_historical_data(api_key, symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
    data = requests.get(url).json()
    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    return df

# Define a function to prepare data for modeling and train a linear regression model
def train_model(df):
    # Create new columns representing lagged prices
    df["sma_5"] = df["4. close"].rolling(window=5).mean()
    df["sma_25"] = df["4. close"].rolling(window=25).mean()
    df[["close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4"]] = df["4. close"].transform(
        lambda x: pd.Series(x.shift(i) for i in range(1, 5)))
    df.dropna(inplace=True)

    # Split data into training and testing sets
    X = df[["4. close", "sma_5", "sma_25", "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4"]]
    y = df["4. close"]
    split_index = int(len(df) * 0.75)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model using r-squared and mean squared error
    r_squared = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    mse = ((y_test - predictions) ** 2).mean()

    print(f"Model trained with r-squared of {r_squared:.4f} and MSE of {mse:.4f}")
    return model

# Define a function to make predictions using the trained model
def make_predictions(api_key, symbol, model):
    # Load the latest data from Alpha Vantage
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&apikey={api_key}"
    data = requests.get(url).json()
    df = pd.DataFrame(data["Time Series (60min)"]).T
    df.index = pd.to_datetime(df.index)

    # Create new columns with lagged prices
    last_price = df["4. close"][-1]
    df["sma_5"] = df["4. close"].rolling(window=5).mean()
    df["sma_25"] = df["4. close"].rolling(window=25).mean()
    df[["close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4"]] = df["4. close"].transform(
        lambda x: pd.Series(x.shift(i) for i in range(1, 5)))
    df.dropna(inplace=True)

    # Prepare data for prediction
    X_pred = df.iloc[[-1]][["4. close", "sma_5", "sma_25", "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4"]]

    # Make prediction with trained model and return result
    prediction = model.predict(X_pred)
    if prediction > last_price:
        return "BUY"
    else:
        return "SELL"

# Example usage
api_key = "YOUR_API_KEY"
symbol = "AAPL"
historical_data = extract_historical_data(api_key, symbol)
model = train_model(historical_data)
prediction = make_predictions(api_key, symbol, model)
print(f"Predicted action for {symbol}: {prediction}")