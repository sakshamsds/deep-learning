# Mega Case Studies - Hybrid Deep Learning Model

# Part 1 - Identifying frauds using Self-Organising Maps

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualizing the results
# Higher the Mean Inter-node Distance, implies more the outlier
from pylab import bone, pcolor, plot, show, colorbar
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,                      #putting marker in the center of winning node
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()  

# Finding the frauds 
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,6)], mappings[(5,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)
frauds = frauds.astype(np.int64)

# Part 2 - Going from Unsupervised to Unsupervised - Detecting the probability of frauds

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Building an ANN
# Import Keras library and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense

# Adding the layers
classifier = Sequential()                                   
classifier.add(Dense(2, kernel_initializer='uniform', activation='relu', input_shape=(15,)))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))   

# Compiling and fitting
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the probabilities of fraud
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1)           #axis=1 horizontal concatenation
y_pred = y_pred[y_pred[:,1].argsort()]
