from math import sqrt
import numpy as np
from collections import Counter
class KNNClassifier():
    def __init__(self, k):
        self._X_train = None
        self._y_train = None
        self.k = k
    def fit(self, X_train, y_train):
        # 没有判断

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        y_predict =  [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    def _predict(self, x):
        """ 单个预测"""
        distances = [sqrt(np.sum(x_train -x )**2) for x_train in self._X_train]
        nearst = np.argsort(distances)
        topk_y = [self._y_train[i] for i in nearst[:self.k]]
        return  Counter(topk_y).most_common(1)[0][0]




