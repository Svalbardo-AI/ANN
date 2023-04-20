import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def deriv_sigmoid(fx):
    return fx*(1-fx)


def cal_number_of_hidden_layer(n):
    return int(np.sqrt(n+1)+2)


class Model:
    def __init__(self, n, m):# n是隐藏层节点数，m是输入节点数
        self.n = n
        self.m = m
        # 创建一级权重矩阵
        cache = []
        for i in range(n):
            cache.append(np.random.randn(m))
        self.w = np.array(cache)

        # 创建一级偏置矩阵
        self.b = np.random.randn(n)

        # 创建二级权重矩阵
        self.v = np.random.randn(n)

        # 输出节点偏置参数
        self.t = np.random.randn(1)

    def train(self, a, epoch, dataframe, y_true):
        if dataframe.shape[1] == self.m:
            print('Initialized, Start training.')
            print('α：{}\tEpoch:{}\t'.format(a, epoch))
            for i in range(epoch):
                for x, y in zip(dataframe, y_true):\
                    # 训练计算
                    h = sigmoid(np.matmul(self.w, x)+self.b)
                    y_pred = sigmoid(np.matmul(self.v, h)+self.t)
                    e = y_pred - y

                    # 计算偏导数
                    d_y_pred = deriv_sigmoid(y_pred)
                    d_h = deriv_sigmoid(h)
                    self.b -= a * e * d_y_pred * self.v * d_h
                    self.t -= a * e * d_y_pred

                    cache1 = np.zeros((self.n, self.m))
                    cache1.transpose()[0] = x[0]
                    cache1.transpose()[1] = x[1]
                    cache2 = np.zeros((self.n, self.m))
                    cache2.transpose()[0] = self.v
                    cache2.transpose()[1] = self.v
                    cache3 = np.zeros((self.n, self.m))
                    cache3.transpose()[0] = d_h
                    cache3.transpose()[1] = d_h
                    self.w -= a * e * d_y_pred * cache1 * cache2 * cache3

                    self.v -= a * e * d_y_pred * h
        else:
            raise ValueError('x 与初始化的变量数量不符')
    def predict(self, x):
        h = sigmoid(np.matmul(self.w, x) + self.b)
        y_pred = sigmoid(np.matmul(self.v, h) + self.t)
        print('y_pred:', y_pred[0])


if __name__ == '__main__':
    data = np.array(
        [[175, 77],
         [-161, -65],
         [-159, -43],
         [175, 92],
         [185, 106],
         [171, 82],
         [-159, -90],
         [-167.8, -78],
         [-150.76, -75]
         ])
    y = np.array(
        [1,
         0,
         0,
         1,
         1,
         1,
         0,
         0,
         0])

    n_input = data.shape[1]
    n_hiden_layer = cal_number_of_hidden_layer(n_input)

    ANN = Model(n_hiden_layer, n_input)
    ANN.train(a=0.1, epoch=3000, dataframe=data, y_true=y)
    ANN.predict([-180,-60])
    ANN.predict([180, 60])
