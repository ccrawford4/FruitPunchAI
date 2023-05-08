# Author: Calum Crawford
# Computer Science Student at the University of San Francisco

# Building a Neural Network from Scratch: 
from sklearn.model_selection import train_test_split # Imports all necessary libraries
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) # Reads in the data 

# Assigns X and y variables
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
y = y.astype('int') # Converts the housing price data into integers 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
X_train.shape # Trains the model


def define_model(input_shape, num_classes): # Adds layers to increase accuracy of the model
    # Build the architecture
    model = keras.Sequential(
        [   keras.layers.Dense(1024, activation="relu", input_shape=input_shape),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(num_classes, activation="linear"),
        ]
    )

    return model

num_epochs = 100
input_shape = X_train[0].shape
num_classes = y_train.shape[-1]

model = define_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

history = model.fit(X_train, y_train, epochs=num_epochs)
y_pred = model.predict(X_test)
MSE = model.evaluate(X_test, y_test) 

# Prints out the neural network Mean Squared Error Value
print(f'This neural network got an MSE score of {MSE[1]}')

# Plotting the data:
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['mse'], label='Training MSE')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss/MSE')
ax.set_title('Training History')
ax.legend()
plt.show()