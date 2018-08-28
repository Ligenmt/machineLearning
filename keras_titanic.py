import urllib.request
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = 'titanic3.xls'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)

all_df = pd.read_excel(filepath)

cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]
# print(filt_df)

def preprocess_data(raw_df):
    # 删除name字段
    df = all_df.drop(['name'], axis=1)
    # 找出含有null值的字段
    # print(all_df.isnull().sum())
    # 将null替换为平均值
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    # print(all_df)

    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype('int')
    # 一位有效编码转换，将embarked转为3个字段
    x_onehot_df = pd.get_dummies(data=df, columns=["embarked"])
    # print(x_onehot_df)

    ndarray = x_onehot_df.values
    # print(ndarray.shape)

    label = ndarray[:, 0]
    features = ndarray[:, 1:]
    # 标准化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_features = minmax_scale.fit_transform(features)
    return scaled_features, label


# 分为测试数据和训练数据
msk = np.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:', len(all_df))
print('train:', len(train_df))
print('test:', len(test_df))

train_features, train_label = preprocess_data(train_df)
test_features, test_label = preprocess_data(test_df)

model = Sequential()
model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=train_features, y=train_label, validation_split=0.1, epochs=40, batch_size=30, verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# show_train_history(train_history, 'acc', 'val_acc')
scores = model.evaluate(x=test_features, y=test_label)
print(scores)