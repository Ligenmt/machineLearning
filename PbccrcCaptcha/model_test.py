from keras.models import *
from keras.layers import *
import random

#测试模型，使用txt文件进行测试

characters = '3456789abcdefghijknpqrstuvxy'
width, height, n_len, n_class = 100, 25, 6, len(characters)

index = 0
def gen(batch_size=32):
    global index
    X = np.zeros((batch_size, height, width, 1), dtype=np.uint8)  # X 的形状是 (batch_size, height, width)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]  # y 的形状是6个 (batch_size, n_class)
    while True:
        index += 1
        index = index % len(data_list)
        for i in range(batch_size):
            captcha = data_list[index]['captcha']
            # img = generator.generate_image(random_str)
            X[i] = data_list[index]['data']
            for j, ch in enumerate(captcha):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

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

# 测试模型
with open('E:\\reportcaptcha\\train_data2.txt', 'r') as f:
    json_str = f.read()

data_list = json.loads(json_str)
random.shuffle(data_list)

test_acc = 0
test_num = 20
for i in range(test_num):
    X_verify, y_verify = next(gen(1))
    y_pred = model.predict(X_verify)
    print('real: %s pred:%s  %s' % (decode(y_verify), decode(y_pred), decode(y_verify) == decode(y_pred)))
    if decode(y_verify) == decode(y_pred):
        test_acc += 1

print('try:%s  hit:%s' % (test_num, test_acc))
