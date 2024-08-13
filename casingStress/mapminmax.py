import numpy as np


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


# 示例
if __name__ == "__main__":
    X = np.array([[45], [30], [45], [60], [45], [30], [45], [90], [75], [30] ])
    X_scaled, min_vals, max_vals = mapminmax(X)
    print("Scaled X:")
    print(X_scaled)
    print("Original Min Values:")
    print(min_vals)
    print("Original Max Values:")
    print(max_vals)