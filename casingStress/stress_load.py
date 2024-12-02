import pandas as pd
import numpy as np
import torch
from casingStress import Net
from casingStress.mapminmax import mapminmax, reverse_mapminmax

model = Net()
model.load_state_dict(torch.load('model_parameters1.pth'))  # 假设模型保存在'model.pth'文件中
# 加载整个模型
# model = torch.load('model1.pth')

model.eval()  # 将模型设置为评估模式

input_data = np.array([30, 20, 30, 60, 2000, 0.1, 20000, 0.1])
input_minmax, x_min, x_max = mapminmax(input_data, [30, 20, 30, 60, 2000, 0.1, 20000, 0.1], [75, 65, 90, 140, 10000, 0.3, 80000, 0.3])
input_minmax = np.array([[np.float32(x) for x in subarr] for subarr in input_minmax], dtype=np.float32)
# 进行预测
with torch.no_grad():  # 在预测时禁用梯度计算
    t_data = torch.tensor(input_minmax, dtype=torch.float32)
    predictions = model(t_data)
    predictions_np = predictions.numpy()

x_denorm = reverse_mapminmax(predictions_np, np.array([23.3619, -83.3194, -32.237]), np.array([1001.05, -8.35487, 14.6481]))
print(x_denorm)