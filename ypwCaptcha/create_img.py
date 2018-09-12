import pydot_ng as pydot
print(pydot.find_graphviz())


from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random


import string
characters = string.digits + string.ascii_uppercase
print(characters)
width, height, n_len, n_class = 170, 80, 4, len(characters)

generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

# plt.imshow(img)
# plt.title(random_str)
# plt.show()


# 无限生成数据函数
def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8) # X 的形状是 (batch_size, height, width, 3)，比如一批生成32个样本，图片宽度为170，高度为80，那么形状就是 (32, 80, 170, 3)，取第一张图就是 X[0]。
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)] # y 的形状是四个 (batch_size, n_class)，如果转换成 numpy 的格式，则是 (n_len, batch_size, n_class)，比如一批生成32个样本，验证码的字符有36种，长度是4位，那么它的形状就是4个 (32, 36)，也可以说是 (4, 32, 36)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            img = generator.generate_image(random_str)
            X[i] = img
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
# print(X.shape)
# print(X[0])
# plt.imshow(X[0])
# plt.title(decode(y))
# plt.show()
# rex = np.reshape(a=X, newshape=(80*170, 3))
# print(rex.shape)
# for i in range(13600):
#     print(rex[i])


from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Conv2D(32*2**i, 3, 3, activation='relu')(x)
    x = Conv2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

print(model.summary())


from keras.utils.vis_utils import plot_model
from IPython.display import Image

plot_model(model, to_file="model.png", show_shapes=True)
Image('model.png')

