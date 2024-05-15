import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load historical data
historical_data_path = "historical_data.csv"
try:
    data = pd.read_csv(historical_data_path)
except FileNotFoundError:
    print(f'Error: Unable to locate file {historical_data_path}')
    exit(1)

def get_features_and_labels(data):
    """Select the features and labels for the predictive model."""
    X = data.drop(['target_asset_price'], axis=1)
    y = data['target_asset_price']
    return X, y

# Split the dataset into training and test sets.
X, y = get_features_and_labels(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model.
model = LinearRegression()
model.fit(X_train, y_train)

# Test the trained model on the test set.
y_pred = model.predict(X_test)
r2metric = r2_score(y_test, y_pred)
print('R2 Metric (closer to 1 is better): %s' % r2metric)

# Generate plot of actual vs. predicted prices
plt.scatter(X_test, y_test, color='purple', label='Actual')
plt.plot(X_test, y_pred, color='green', label='Predicted')
plt.xlabel('Features')
plt.ylabel('Target Asset Price')
plt.title('Actual vs. Predicted Target Asset Price')
plt.legend()
plt.show()

# Suggestions implemented:
# 1. Added necessary imports for train_test_split, LinearRegression.
# 2. Added try-except block to handle file not found error.
# 3. Added comments to explain the purpose of each section of the code and any important details to understand.
# 4. Generated a plot to visualize the results of the example predictive model to identify potential issues and improvements for the trading strategy.
