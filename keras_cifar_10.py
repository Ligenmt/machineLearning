from keras.datasets import cifar10
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)


(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()

print(x_train_image.shape)
print(x_test_image.shape)
print(y_train_label.shape)

label_dict = {0: "plane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)  #建立subgraph子图形为5行5列
        ax.imshow(images[idx], cmap='binary')  #画出subgraph子图形
        title = str(i) + label_dict[labels[i][0]]  #设置子图形title
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[idx]]
        ax.set_title(title, fontsize=10) #设置子标题
        ax.set_xticks([])
        ax.set_yticks([])  #设置不显示刻度
        idx += 1
    plt.show()

# 每一点都由三原色组成
print(x_train_image[0][0][0])
# plot_images_labels_prediction(x_train_image, y_train_label, [], 0)

# 标准化
x_train_image_normalize = x_train_image.astype('float32') / 255.0
x_test_image_normalize = x_test_image.astype('float32') / 255.0
# 将label标签字段转换为一位有效编码
y_train_label_onehot = np_utils.to_categorical(y_train_label)
y_test_label_onehot = np_utils.to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


model = Sequential()
# 卷积层1
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(rate=0.25))
# 池化层1
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷积层2
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(rate=0.25))
# 池化层2
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷积层3
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(rate=0.3))
# 池化层3
model.add(MaxPooling2D(pool_size=(2, 2)))

# 平坦层
model.add(Flatten())
model.add(Dropout(rate=0.25))
# 隐藏层
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
# 输出层
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# try:
#     model.load_weights('cifarCnnModel.h5')
# except:
#     print('加载模型失败!')
train_history = model.fit(x_train_image_normalize, y_train_label_onehot, validation_split=0.2, epochs=30, batch_size=128, verbose=2)
model.save_weights('cifarCnnModel.h5')

