from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 假设我们有一组数据
# 注意：在实际应用中，X通常是一个二维数组（或DataFrame），其中每一行代表一个样本，每一列代表一个特征
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# 实例化MinMaxScaler对象
# 默认情况下，feature_range=(0, 1)，即数据将被缩放到[0, 1]的范围内
scaler = MinMaxScaler(feature_range=(-1, 1))

# 拟合并转换数据
# 注意：fit_transform方法会同时计算缩放参数并转换数据
X_scaled = scaler.fit_transform(X)

print("原始数据:")
print(X)
print("\n缩放后的数据:")
print(X_scaled)

# 如果你有新的数据需要按照相同的规则进行缩放
X_new = np.array([[0, 2],
                  [6, 8]])

# 使用transform方法进行缩放
# 注意：transform方法会使用之前fit_transform方法计算得到的缩放参数
X_new_scaled = scaler.transform(X_new)

print("\n新数据:")
print(X_new)
print("\n新数据缩放后:")
print(X_new_scaled)

# 如果你需要将缩放后的数据转换回原始尺度（反归一化）
X_original = scaler.inverse_transform(X_scaled)

print("\n缩放后数据反转换回原始数据:")
print(X_original)