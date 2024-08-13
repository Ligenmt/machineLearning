import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def mapminmax(X, ymin=-1, ymax=1):
    """
    将矩阵X的每个特征缩放到[ymin, ymax]范围内。

    参数:
    X -- 输入的numpy数组或矩阵
    ymin -- 缩放后的最小值，默认为-1
    ymax -- 缩放后的最大值，默认为1

    返回:
    X_scaled -- 缩放后的numpy数组
    min_vals -- 原始数据中的最小值
    max_vals -- 原始数据中的最大值
    """
    # 确保X是二维的，如果不是，则增加一个维度
    if X.ndim == 1:
        X = X.reshape(1, -1)

        # 计算每列的最小值和最大值
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # 缩放数据
    X_scaled = (X - min_vals) * (ymax - ymin) / (max_vals - min_vals) + ymin

    # 替换0除法的情形（如果某一列全为相同的值）
    # np.place(X_scaled, (max_vals - min_vals) == 0, ymin)

    return X_scaled, min_vals, max_vals


file_path = "C:\\Users\\ligen\\Documents\\MATLAB\\wellintegrity\\10_套管-水泥环应力模型训练\\套管应力-水泥环应力模型训练数据集20231118 - 副本.xlsx"
df = pd.read_excel(file_path, sheet_name='2')
print(df.shape)
# 显示DataFrame的前几行来验证数据是否正确读取
# print(df.head())

df1 = df.iloc[1:150001, 0:11]

df11 = df.iloc[200000:, 0:11]

print(df1.shape, df11.shape)

data = pd.concat([df1, df11], axis=0)

print(data.shape)

# print(len(data1))
idx = 0
for row in data.values:
    for ele in row:
        print(ele, end=' ')
    print()
    idx += 1
    if idx > 10:
        break

data = data.transpose()

nn = data.shape[1]

line1 = 0  # 输入变量起始行数
line2 = 8  # 输入变量终止行数
n_line = line2 - line1   # 输入特征个数
input = data.iloc[line1:line2, :]
output = data.iloc[line2:, :]

n_train = round(nn * 0.8)
n_valid = round(nn * 0.1)
n_test = nn - n_train - n_valid

# input_train = np.zeros((n_line, n_train))
# output_train = np.zeros((3, n_train))
# input_valid = np.zeros((n_line, n_valid))
# output_valid = np.zeros((3, n_valid))
# input_test = np.zeros((n_line, n_test))
# output_test = np.zeros((3, n_test))

input_train = input.iloc[:, :n_train]
output_train = output.iloc[:, :n_train]

print(input_train.shape, output_train.shape)

input_train_value = input_train.values
output_train_value = output_train.values

input_train_trans_value = input_train.transpose().values
output_train_trans_value = output_train.transpose().values

# scaler1 = MinMaxScaler()
# inputn = scaler1.fit_transform(input_train_value)

# scaler2 = MinMaxScaler()
# inputn2 = scaler2.fit_transform(input_train_trans_value)

inputn, min_vals, max_vals = mapminmax(input_train_trans_value)
print("Scaled inputn:")
print(inputn)
print("Original Min Values:")
print(min_vals)
print("Original Max Values:")
print(max_vals)

outputn, _, _ = mapminmax(output_train_trans_value)

hidden_layer_neures=40




print()

