from keras.models import *
from keras.layers import *

import numpy as np
import base64
from io import BytesIO
from PIL import Image
# 训练好的模型，可用于预测 pbccrc_captcha_model.model.predict(X)

characters = '3456789abcdefghijknpqrstuvxy'
width, height, n_len, n_class = 100, 25, 6, len(characters)


def image_to_input(path):
    image = Image.open(path)
    image_L = image.convert('1')   #转化为二值化图
    img_binary = np.array(image_L)
    (row, col) = img_binary.shape
    img_binary_list = img_binary.tolist()
    X = np.zeros((1, height, width, 1), dtype=np.uint8)  # X 的形状是 (batch_size, height, width)
    for r in range(row):
        for c in range(col):
            value = img_binary_list[r][c]
            if value:
                img_binary_list[r][c] = [1]
            else:
                img_binary_list[r][c] = [0]
    X[0] = img_binary_list
    return X

def base64_to_input(img_base64):
    X = np.zeros((1, height, width, 1), dtype=np.uint8)  # X 的形状是 (batch_size, height, width)
    # y = [np.zeros((1, n_class), dtype=np.uint8) for i in range(n_len)]  # y 的形状是6个 (batch_size, n_class)
    pfd = BytesIO(base64.b64decode(img_base64))
    img = Image.open(pfd)
    L = img.convert('1')   #转化为二值化图
    img_binary = np.array(L)
    (row, col) = img_binary.shape
    img_binary_list = img_binary.tolist()
    for r in range(row):
        for c in range(col):
            value = img_binary_list[r][c]
            if value:
                img_binary_list[r][c] = [1]
            else:
                img_binary_list[r][c] = [0]
    X[0] = img_binary_list
    return X

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])

input_tensor = Input((height, width, 1))
x = input_tensor
for i in range(2):
    x = Conv2D(filters=16*2**i, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=16*2**i, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(6)]
model = Model(inputs=input_tensor, outputs=x)

model.load_weights('pbccrc_captcha_model.h5')