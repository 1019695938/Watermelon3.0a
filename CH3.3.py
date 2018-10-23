import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def sigmoid(x):
    y = 1.0 / (1 + np.exp(-x))     # Sigmoid函数
    return y


def newton(X, y):     # 牛顿法
    N = X.shape[0]
    beta = np.ones((1, 3))     # 设定初值
    # shape [N, 1]
    z = X.dot(beta.T)
    # 最大化对数似然
    old_l = 0
    new_l = np.sum(y * z + np.log(1 + np.exp(z)))
    iters = 0    # 迭代次数
    while np.abs(old_l - new_l) > 1e-5:
        # shape [N, 1]
        p1 = np.exp(z) / (1 + np.exp(z))     # p(y=1|x)
        # shape [N, N]
        p = np.diag((p1 * (1 - p1)).reshape(N))     # p(y=0|x)
        # shape [1, 3]
        first_order = -np.sum(X * (y - p1), 0, keepdims=True)
        # shape [3, 3]
        second_order = X.T.dot(p).dot(X)

        # 迭代更新公式
        beta -= first_order.dot(np.linalg.inv(second_order))
        z = X.dot(beta.T)
        old_l = new_l
        new_l = np.sum(y * z + np.log(1 + np.exp(z)))

        iters += 1
    print("iters:", iters)
    print(new_l)
    return beta


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv("watermelon_3.0a.csv", header=None)
    data.insert(3, "3", 1)
    X = data.values[:, 1:-1]
    y = data.values[:, 4].reshape(-1, 1)

    # 分割训练数据(区分标签)
    positive_data = data.values[data.values[:, 4] == 1.0, :]     #好瓜
    negative_data = data.values[data.values[:, 4] == 0, :]     #坏瓜
    plt.plot(positive_data[:, 1], positive_data[:, 2], 'bo')
    plt.plot(negative_data[:, 1], negative_data[:, 2], 'r+')

    # 牛顿法
    beta = newton(X, y)
    newton_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    newton_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [newton_left, newton_right], 'y-')

    plt.xlabel('density')     # X轴:密度
    plt.ylabel('sugar rate')     # Y轴:含糖率
    plt.title("Rate regression")
    plt.show()
