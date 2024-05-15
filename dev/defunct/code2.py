import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load historical data
# Replace the path here with the path to your own historical data.
data = pd.read_csv("historical_data.csv")

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