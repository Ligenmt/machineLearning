from keras.datasets import cifar10

import numpy as np

np.random.seed(10)


(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()

print(x_train_image.shape)
print(x_test_image.shape)
print(y_train_label.shape)

label_dict = {0: "飞机", 1: "汽车", 2: "鸟", 3: "猫", 4: "鹿", 5: "狗", 6: "青蛙", 7: "马", 8: "船", 9: "卡车"}
