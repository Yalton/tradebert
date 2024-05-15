import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# Define ticker symbol
ticker = "AAPL"

# Define date range
start_date = "2016-01-01"
end_date = "2021-12-31"

# Fetch data from yahoo finance API
data = yf.download(ticker, start=start_date, end=end_date)

# Extract the Close column
close_data = data[["Close"]].values

# Fit an HMM model
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(close_data)

# Predict hidden states
hidden_states = model.predict(close_data)

# Print state transitions
print("Transition matrix")
print(model.transmat_)

# Print the means and covariances of each state
for i in range(model.n_components):
    print(f"Mean of state {i+1}: {model.means_[i][0]:.2f}")
    print(f"Covariance of state {i+1}:")
    print(model.covars_[i])
    print()