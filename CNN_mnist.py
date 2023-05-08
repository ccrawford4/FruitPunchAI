# Author: Calum Crawford
# Computer Science Student at the University of San Francisco

# Creating and tuning a convolutional neural network to evaluate the mnist dataset:

# Imports all the necessary libraries and classes
from keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization, Activation)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data() # Trains the data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_epochs = 10
batch_size = 32
input_shape = X_train.shape[1:]
num_classes = y_train.shape[-1]



def define_cnn(input_shape, num_classes): # Defines the CNN
    model = keras.Sequential()

    # first CONV => RELU => CONV => RELU 
    model.add(Conv2D(32, (2, 2), padding="same", # Adds layers to increase accuracy of model
        input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    # second CONV => RELU => CONV => RELU
    
    model.add(Conv2D(64, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax")) 

    model.summary()
    return model



modelA = define_cnn(input_shape, num_classes)
modelA.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# Train and test
H = modelA.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test))
score = modelA.evaluate(X_test, y_test, verbose=0)
print("MNIST DATA: ") # Prints out the amount lost and the accuracy of the model at the end
print("Test loss:", score[0])
print("Test accuracy:", score[1])

fig, ax = plt.subplots()
ax.plot(H.history['loss'], label='Training Loss')
ax.plot(H.history['accuracy'], label='Training Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training History')
ax.legend()
plt.show()
