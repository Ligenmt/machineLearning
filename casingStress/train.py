import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 3)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        # softmax做归一化
        # x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        # x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)
        return x


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

input_train = input.iloc[:, :n_train]
output_train = output.iloc[:, :n_train]

print(input_train.shape, output_train.shape)

input_train_value = input_train.values
output_train_value = output_train.values
input_train_trans_value = input_train.transpose().values
output_train_trans_value = output_train.transpose().values

print(type(input_train_trans_value))
print(input_train_trans_value.shape)

# scaler1 = MinMaxScaler()
# inputn = scaler1.fit_transform(input_train_value)

# scaler2 = MinMaxScaler()
# inputn2 = scaler2.fit_transform(input_train_trans_value)

inputn, input_min_vals, input_max_vals = mapminmax(input_train_trans_value)
print("Scaled inputn:")
print(inputn)
print("Original Min Values:")
print(input_min_vals)
print("Original Max Values:")
print(input_max_vals)

"""
Scaled inputn:
[[-1.0 -1.0 -1.0 ... -1.0 -1.0 -1.0]
 [-1.0 -1.0 -1.0 ... -1.0 -1.0 -0.5]
 [-1.0 -1.0 -1.0 ... -1.0 -1.0 2.220446049250313e-16]
 ...
 [1.0 -1.0 0.5 ... 1.0 1.0 2.220446049250313e-16]
 [1.0 -1.0 0.5 ... 1.0 1.0 0.5]
 [1.0 -1.0 0.5 ... 1.0 1.0 1.0]]
Original Min Values:
[30 20 30 60 2000 0.1 20000 0.1]
Original Max Values:
[90 65 90 140 10000 0.3 80000 0.3]
"""

outputn, output_min_vals, output_max_vals = mapminmax(output_train_trans_value)
print("Scaled outputn:")
print(outputn)
print("Original Min Values:")
print(output_min_vals)
print("Original Max Values:")
print(output_max_vals)

model = Net()
criterion = torch.nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练数据示例
# 假设有100个样本，每个样本有8个特征，标签有3个输出
# train_data = torch.randn(100, 8)
# train_labels = torch.randn(100, 3)

print(type(inputn))
print(inputn.shape)
inputn = np.array([[np.float32(x) for x in subarr] for subarr in inputn], dtype=np.float32)
outputn = np.array([[np.float32(x) for x in subarr] for subarr in outputn], dtype=np.float32)

train_data = torch.tensor(inputn, dtype=torch.float32)
train_labels = torch.tensor(outputn, dtype=torch.float32)


num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成")
torch.save(model.state_dict(), 'model_parameters1.pth')
torch.save(model, 'model1.pth')



