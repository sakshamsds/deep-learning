# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
training_set = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:, 1:2].values                                  #we need matrix not vector

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

'''
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
'''

# Getting the inputs and outputs
X_train = training_set_scaled[0:1257,:]
y_train = training_set_scaled[1:1258,:]

# Reshape
X_train = np.reshape(X_train, (1257, 1, 1))                                      #(observations, timesteps, features)

# Part 2 - Building the RNN

# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initializing RNN
regressor = Sequential()

# Adding the input layer and LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(X_train.shape[1], 1)))

# Adding the output layer
regressor.add(Dense(units=1))                                                   #Stock price at t+1

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')                  #generally RMSprop used in RNN

# Fitting the RNN to training_set
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

# Part 3 - Making the predicitons and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = test_set.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)                                                   # Scaling the inputs
inputs = np.reshape(inputs, (20, 1, 1))                                         # reshaping
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

'''
# Getting the predicted stock price of 2017
scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(1258, 1278):
    inputs.append(scaled_real_stock_price[i-60:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
'''

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

'''
# TODO
# Getting the real stock price of 2012-2016
real_stock_price_train = pd.read_csv("Google_Stock_Price_Train.csv")
real_stock_price_train = real_stock_price_train.iloc[:, 1:2].values   

# Getting the predicted stock price of 2012-2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualising the results
plt.plot(real_stock_price_train, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_train, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
'''

# Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
rmse = rmse*100/800                                                     #rmse in %, avg stock price = 800
