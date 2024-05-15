from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict the values for X_test
y_pred = self.model.predict(X_test)

# Denormalize the predicted and actual values if needed
# y_pred = scaler.inverse_transform(y_pred)
# Y_test = scaler.inverse_transform(Y_test)

# Calculate metrics
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R^2 Score:', r2)