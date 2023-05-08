# Author: Calum Crawford
# Computer Science Student at the University of San Francisco

# Using a CNN to evaluate the fasion mnist dataset

from CNN_mnist import define_cnn # Gets the define_cnn method from the other python file

# Imports necessary libraries and other packages
from keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization, Activation)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() # Trains the data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_epochs = 10
batch_size = 32
input_shape = X_train.shape[1:]
num_classes = y_train.shape[-1]

modelB = define_cnn(input_shape, num_classes)
modelB.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# Train and test
H = modelB.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test))
score = modelB.evaluate(X_test, y_test, verbose=0)
print("FASHION MNIST DATA: ") # Prints out the test loss and accuracy of the model on the fashion mnist data set
print("Test loss:", score[0])
print("Test accuracy:", score[1])

fig, ax = plt.subplots()
ax.plot(H.history['loss'], label='Training Loss')
ax.plot(H.history['accuracy'], label='Training Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss/MSE')
ax.set_title('Training History')
ax.legend()
plt.show()