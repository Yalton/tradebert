# def train(self, data): 

#         # preprocess the data
#         data, _ = preProcess(data)

#         X = data.drop(['Close'], axis=1)
#         y = data['Close']

#         # Fit the scaler to the 'Close' column
#         self.scaler = MinMaxScaler().fit(y.values.reshape(-1, 1))

#         # Transform the 'Close' column
#         y = self.scaler.transform(y.values.reshape(-1, 1))

#         X_train, X_test, Y_train, Y_test = train_test_split(X, y)

#         self.model.add(Dense(1, input_shape=(X_train.shape[1],), activation='tanh'))
#         self.model.add(Dense(3, activation='tanh'))
#         self.model.add(Dense(3, activation='tanh'))
#         self.model.add(Dense(3, activation='tanh'))
#         self.model.add(Dense(1, input_shape=(X_train.shape[1],), activation='linear'))  # changed activation to 'linear'


#         self.model.compile(
#                     optimizer='rmsprop',
#                     loss='mse',  # changed loss to 'mse'
#                     metrics=['accuracy']
#                     )

#         X_train = X_train.astype('float32')
#         Y_train = Y_train.astype('float32')
#         self.model.fit(X_train, Y_train, epochs=100)

#         # Evaluate the model
#         test_loss, test_mae = self.model.evaluate(X_test, Y_test)

#         print('Test loss:', test_loss)
#         print('Test MAE:', test_mae)

#         self.model.save('models/LinearRegressor.h5')

#     def predict(self, X):
#         self.model = load_model('models/LinearRegressor.h5')
        
#         X = X.select_dtypes(exclude=['object'])
#         # Convert X to a numpy array of type float32
#         X = X.values.astype('float32')
        
#         predictions = self.model.predict(X)

#         # Apply inverse transformation to your predictions
#         predictions = self.scaler.inverse_transform(predictions)

#         return predictions


# def train(self, data): 
    
#     data = preProcess(data)

#     prices = data['Close'].values.reshape(-1, 1)

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     prices = scaler.fit_transform(prices)

#     look_back = 10
#     X, Y = self.create_dataset(prices, look_back)

#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     self.model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     self.model.add(Dropout(0.2))
#     self.model.add(SimpleRNN(units=50))
#     self.model.add(Dropout(0.2))
#     self.model.add(Dense(1))

#     self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
#     self.model.fit(X_train, Y_train, epochs=100, batch_size=32)

#     # Evaluate the model
#     test_loss, test_mae = self.model.evaluate(X_test, Y_test)

#     print('Test loss:', test_loss)
#     print('Test MAE:', test_mae)

#     self.model.save('models/RNN.h5')


    # y_pred = self.y_scaler.inverse_transform(y_pred)

    # mse = mean_squared_error(Y_test, y_pred)
    # mae = mean_absolute_error(Y_test, y_pred)
    # r2 = r2_score(Y_test, y_pred)

    # print('Mean Squared Error:', mse)
    # print('Mean Absolute Error:', mae)
    # print('R^2 Score:', r2)

    # #y_pred = self.y_scaler.inverse_transform(y_pred)

    # print("Inverse Transformed y_pred", y_pred)


        # if isinstance(X, pd.DataFrame):
    #     # Ensure that the input X also drops the 'Close' and 'Adj Close' columns
    #     if 'Close' in X.columns:
    #         X = X.drop(['Close', 'Adj Close'], axis=1)
    #     X = X.select_dtypes(exclude=['object'])
    #     X = self.x_scaler.transform(X)
    # elif isinstance(X, np.ndarray):
    #     X = X.reshape((X.shape[0], 1, X.shape[1]))

    # predictions = self.model.predict(X)
    # predictions = self.y_scaler.inverse_transform(predictions)

    # return predictions


# def preProcess(data):
#     # Check for NaN values
#     print(data.isnull().sum())

#     # Drop rows with NaN values
#     data = data.dropna()

#     # Create a scaler
#     scaler = MinMaxScaler(feature_range=(0, 1))

#     # Only include numerical columns in the data that you pass to the scaler
#     numerical_columns = data.select_dtypes(include=[np.number]).columns

#     # Fit the scaler to your data and transform it
#     data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

#     data = data.select_dtypes(exclude=['object'])

#     # Return the preprocessed data AND the fitted scaler
#     return data, scaler


# def train(self, data): 

#     data, self.scaler = preProcess(data)

#     X = data.drop(['Close'], axis=1)
#     y = data['Close']

# def train(self, data): 

#         # preprocess the data
#         data, _ = preProcess(data)

#         X = data.drop(['Close', 'Adj Close'], axis=1)  # drop 'Adj Close'
#         y = data['Close']

#         # Fit the scaler to the 'Close' column
#         self.scaler = MinMaxScaler().fit(y.values.reshape(-1, 1))

#         # Transform the 'Close' column
#         y = self.scaler.transform(y.values.reshape(-1, 1))

# data = preProcess(data)

# X = data.drop(['Close', 'Adj Close'], axis=1)  
# y = data['Close']

# # Fit the scaler to the 'Close' column
# self.y_scaler = MinMaxScaler().fit(y.values.reshape(-1, 1))

# # Transform the 'Close' column
# y = self.y_scaler.transform(y.values.reshape(-1, 1))

# # Fit the scaler to the X data
# self.x_scaler = MinMaxScaler().fit(X)

# # Transform the X data
# X = self.x_scaler.transform(X)

# look_back = 10

# X, Y = self.create_dataset(X, look_back)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))


# # self.model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# # self.model.add(Dropout(0.3))  # Increased dropout
# # self.model.add(SimpleRNN(100, return_sequences=True))
# # self.model.add(Dropout(0.3))  # Increased dropout
# # self.model.add(SimpleRNN(100, return_sequences=True))  # Extra layer
# # self.model.add(Dropout(0.3))  # Increased dropout
# # self.model.add(SimpleRNN(50))  # Extra layer
# # self.model.add(Dropout(0.3))  # Increased dropout
# # self.model.add(Dense(1))

# # self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
# l2_reg = 0.001

# self.model.add(SimpleRNN(50, return_sequences=True, kernel_regularizer=l2(l2_reg), input_shape=(X_train.shape[1], X_train.shape[2])))
# self.model.add(Dropout(0.3))  
# self.model.add(SimpleRNN(100, return_sequences=True, kernel_regularizer=l2(l2_reg)))
# self.model.add(Dropout(0.3))  
# self.model.add(SimpleRNN(100, return_sequences=True, kernel_regularizer=l2(l2_reg)))
# self.model.add(Dropout(0.3)) 
# self.model.add(SimpleRNN(50, kernel_regularizer=l2(l2_reg)))
# self.model.add(Dropout(0.3)) 
# self.model.add(Dense(1))

# # Initial learning rate
# lr_initial = 0.01
# # The decay steps
# decay_steps = 10000
# # The base of the exponential
# lr_decay_rate = 0.96

# lr_schedule = ExponentialDecay(lr_initial, decay_steps=decay_steps, decay_rate=lr_decay_rate, staircase=True)
# optimizer = Adam(learning_rate=lr_schedule)

# self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
# self.model.fit(X_train, Y_train, epochs=100, batch_size=32)

# y_pred = self.model.predict(X_test)

# y_pred_unscaled = self.y_scaler.inverse_transform(y_pred)
# Y_test_unscaled = self.y_scaler.inverse_transform(Y_test.reshape(-1, 1))  # Unscale the true target values

# mse = mean_squared_error(Y_test_unscaled, y_pred_unscaled)
# mae = mean_absolute_error(Y_test_unscaled, y_pred_unscaled)
# self.r2 = r2_score(Y_test_unscaled, y_pred_unscaled)

# print('Mean Squared Error:', mse)
# print('Mean Absolute Error:', mae)
# print('R^2 Score:', self.r2)