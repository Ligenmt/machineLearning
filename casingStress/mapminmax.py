import numpy as np


def mapminmax(X, global_min=None, global_max=None, feature_range=(-1, 1)):
    """
    将矩阵X的每个特征缩放到feature_range范围内。

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
    if global_min is None:
        min_vals = X.min(axis=0)
    else:
        min_vals = np.array(global_min)
    if global_max is None:
        max_vals = X.max(axis=0)
    else:
        max_vals = np.array(global_max)
    ymin = feature_range[0]
    ymax = feature_range[1]

    # 缩放数据
    X_scaled = (X - min_vals) * (ymax - ymin) / (max_vals - min_vals) + ymin

    # 替换0除法的情形（如果某一列全为相同的值）
    # np.place(X_scaled, (max_vals - min_vals) == 0, ymin)

    return X_scaled, min_vals, max_vals


def reverse_mapminmax(X_norm, X_min, X_max, feature_range=(-1, 1)):
    """
    反归一化二维数组X_norm。

    参数:
    X_norm: 归一化后的二维数组。
    X_min: 每列的最小值。
    X_max: 每列的最大值。
    feature_range: 归一化的范围，默认为(-1, 1)。
    返回:
    X: 反归一化后的数组。
    """
    X_range = X_max - X_min
    X = (X_norm - feature_range[0]) / (feature_range[1] - feature_range[0]) * X_range + X_min
    return X

# 示例
if __name__ == "__main__":
    X = np.array([[45, 20], [30, 20], [45, 20], [60, 20], [45, 30], [30, 30], [45, 30], [90, 30], [75, 40], [30, 40]])
    X_scaled, x_min, x_max = mapminmax(X)
    print("Scaled X:\n", X_scaled)
    print("Original Min Values:", x_min)
    print("Original Max Values:", x_max)
    x_denorm = reverse_mapminmax(X_scaled, x_min, x_max)
    # x_denorm = reverse_mapminmax(np.array([-0.5, -1]), x_min, x_max)
    print("Denormalized data:\n", x_denorm)

    X_scaled, x_min, x_max = mapminmax(np.array([45, 20]), [30, 20], [90, 40])
    print("Scaled X:", X_scaled)