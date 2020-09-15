import numpy as np
def train_test_split(X, y,test_radio=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    # 生成随机索引
    shuffled_index = np.random.permutation(len(X))
    test_size = int(len(X)*test_radio)
    test_indexs = shuffled_index[:test_size]
    train_indexs = shuffled_index[test_size:]
    X_train =X[train_indexs]
    y_train = y[train_indexs]

    X_test = X[test_indexs]
    y_test = y[test_indexs]
    return X_train, X_test, y_train, y_test

