import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def LR(X0, X1):

    #shape [1, d]
    mean0 = np.mean(X0, axis=0, keepdims=True)     #压缩行，对各列求均值，返回 1* d 矩阵
    mean1 = np.mean(X1, axis=0, keepdims=True)     #压缩行，对各列求均值，返回 1* d 矩阵
    Ew = (X0-mean0).T.dot(X0-mean0) + (X1-mean1).T.dot(X1-mean1)
    omega = np.linalg.inv(Ew).dot((mean0-mean1).T)
    return omega


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv("watermelon_3.0a.csv", header=None)
    positive_data = data.values[data.values[:, -1] == 1.0, :]
    negative_data = data.values[data.values[:, -1] == 0.0, :]
    print(positive_data)

    # 线性回归公式
    omega = LR(negative_data[:, 1:-1], positive_data[:, 1:-1])

    # 分割训练数据(区分标签)
    plt.plot(positive_data[:, 1], positive_data[:, 2], "bo")
    plt.plot(negative_data[:, 1], negative_data[:, 2], "r+")
    lda_left = 0
    lda_right = -(omega[0]*0.9) / omega[1]
    plt.plot([0, 0.9], [lda_left, lda_right], 'y-')

    plt.xlabel('density')     # X轴:密度
    plt.ylabel('sugar rate')     # Y轴:含糖率
    plt.title("Linear regression")
    plt.show()