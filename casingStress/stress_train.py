import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from casingStress import Net
from casingStress.mapminmax import mapminmax


file_path = "套管应力-水泥环应力模型训练数据集20231118 - 副本.xlsx"
df = pd.read_excel(file_path, sheet_name='2')
print(df.shape)
# 显示DataFrame的前几行来验证数据是否正确读取
# print(df.head())

data = df.iloc[1:, 0:11]
input_data = df.iloc[1:, 0:8]
output_data1 = df.iloc[1:, 8:11].transpose()
output_data2 = df.iloc[1:, 11:14].transpose()
output_data3 = df.iloc[1:, 14:17].transpose()
output_data4 = df.iloc[1:, 17:20].transpose()
output_data5 = df.iloc[1:, 20:23].transpose()
output_data6 = df.iloc[1:, 23:26].transpose()

print(data.shape)

data = data.transpose()

nn = data.shape[1]

line1 = 0  # 输入变量起始行数
line2 = 8  # 输入变量终止行数
input = data.iloc[line1:line2, :]
output = data.iloc[line2:, :]
output = output_data2

n_train = round(nn * 0.8)
n_valid = round(nn * 0.1)
n_test = nn - n_train - n_valid

input_train = input.iloc[:, :n_train]
output_train = output.iloc[:, :n_train]

input_valid = input.iloc[:, n_train:n_train + n_valid]
output_valid = output.iloc[:, n_train:n_train + n_valid]

input_test = input.iloc[:, n_train + n_valid:]
output_test = output.iloc[:, n_train + n_valid:]

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
print("Input Original Min Values:")
print(input_min_vals)
print("Input Original Max Values:")
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
print("Output Original Min Values:")
print(output_min_vals)
print("Output Original Max Values:")
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

torch.save(model.state_dict(), 'model_parameters2.pth')
torch.save(model, 'model2.pth')
print("保存模型参数")
# test

input_data = np.array([30, 20, 30, 60, 2000, 0.1, 20000, 0.1])
input_minmax, x_min, x_max = mapminmax(input_data, [30, 20, 30, 60, 2000, 0.1, 20000, 0.1], [75, 65, 90, 140, 10000, 0.3, 80000, 0.3])
input_minmax = np.array([[np.float32(x) for x in subarr] for subarr in input_minmax], dtype=np.float32)
t_data = torch.tensor(input_minmax, dtype=torch.float32)
predictions = model(t_data)
predictions_np = predictions.detach().numpy()
print("第一行数据预测结果", predictions_np)




