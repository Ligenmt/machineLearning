import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Keras多层感知机识别手写数字

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)  #设置显示图形的大小
    plt.imshow(image, cmap='binary')  #显示图形，binary以黑白灰度显示
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)  #建立subgraph子图形为5行5列
        ax.imshow(images[idx], cmap='binary')  #画出subgraph子图形
        title = 'label=' + str(labels[idx])  #设置子图形title
        if len(prediction) > 0:
            title += ', predict=' + str(prediction[idx])
        ax.set_title(title, fontsize=10) #设置子标题
        ax.set_xticks([])
        ax.set_yticks([])  #设置不显示刻度
        idx += 1
    plt.show()
#
#
# # plot_image(x_train_image[0])
# plot_images_labels_prediction(x_test_image, y_test_label, [], 0, 10)

# 图像转为一维向量
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
print('x_train:', x_Train.shape)
print('x_test:', x_Test.shape)

# 将features标准化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
# print(x_Test_normalize[0])

# 将labels进行One-Hot Encoding转换
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
print(y_train_label[:6])
print(y_TrainOneHot[:6])

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 建立线性堆叠模型
model = Sequential()
# 建立输入层和隐藏层
model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
# 在隐藏层加入Dropout功能可以改善过拟合
model.add(Dropout(0.5))

# 加入新的隐藏层
model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))

# 建立输出层(keras会自动按照上一层units设置这一层的input_dim)
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
# print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_Train_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')

scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print()
print('accuracy:', scores)

# 进行预测
prediction = model.predict_classes(x_Test)
print(prediction)

# plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340)

# 显示混淆矩阵
# ct = pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
# print(ct)