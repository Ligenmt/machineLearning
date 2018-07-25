from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(10)

# Keras卷积神经网络识别手写数字

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

x_train4d = x_train_image.reshape(x_train_image.shape[0], 28, 28, 1).astype('float32')
x_test4d = x_test_image.reshape(x_test_image.shape[0], 28, 28, 1).astype('float32')

# print(x_train_image[0])
# print(x_train4d.shape)
# print(x_test4d.shape)

x_train4d_normalize = x_train4d / 255
x_test4d_normalize = x_test4d / 255

# one_hot_encoding转换
y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot = np_utils.to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_train4d_normalize, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=300, verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'acc', 'val_acc')