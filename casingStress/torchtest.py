import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # 输入8个参数，隐藏层16个神经元
        self.fc2 = nn.Linear(16, 3)  # 隐藏层输出3个参数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 假设你的训练数据为以下 NumPy 数组
numpy_train_data = np.random.rand(100, 8)  # 示例输入数据，形状为 (100, 8)
numpy_train_labels = np.random.rand(100, 3)  # 示例标签数据，形状为 (100, 3)
print(numpy_train_data.shape)
print(type(numpy_train_data))
# 将 NumPy 数组转换为 PyTorch 张量
train_data = torch.tensor(numpy_train_data, dtype=torch.float32)
train_labels = torch.tensor(numpy_train_labels, dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成")