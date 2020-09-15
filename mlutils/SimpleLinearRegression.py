import numpy as np
class SimpleLinearRegression1():
    def __init__(self):
        # 变量
        self.a_ = None
        self.b_ = None
    def fit(self, X_train, y_train):
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        # 分子
        num = 0.0
        # 分母
        d = 0.0
        for x_i, y_i in zip(X_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        self.a_ = num/d
        self.b_ = y_mean - self.a_*x_mean
        return self
    def predict(self,x_predict):
        return np.array([self._predict(x) for x in x_predict])
    def _predict(self, x_single):
        return self.a_*x_single +self.b_
    def __repr__(self):
        return "SimpleLinearRegression1()"

class SimpleLinearRegression2():
    def __init__(self):
        # 变量
        self.a_ = None
        self.b_ = None
    def fit(self, X_train, y_train):
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        # 分子
        num = 0.0
        # 分母
        d = 0.0
        # for x_i, y_i in zip(X_train, y_train):
        #     num += (x_i - x_mean) * (y_i - y_mean)
        #     d += (x_i - x_mean) ** 2
        self.a_ = (X_train-x_mean).dot(y_train-y_mean)/(X_train-x_mean).dot((X_train-x_mean))
        self.b_ = y_mean - self.a_*x_mean
        return self
    def predict(self,x_predict):
        return np.array([self._predict(x) for x in x_predict])
    def _predict(self, x_single):
        return self.a_*x_single +self.b_
    def __repr__(self):
        return "SimpleLinearRegression1()"