import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection

# 15% for validation and test, 70% for training in total
X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5
)

print(X_train.shape, y_train.shape)


class LinearRegression():
    def __init__(self, n_features):
        np.random.seed(10)
        self.W = np.random.randn(n_features,1)
        self.b = np.random.randn(1)
        

    def preditctions(self, X):
        # y_prediction = (X*self.W) + self.b
        # return y_prediction
        pass
    def fit_model():
        pass

    def _update_params(self, new_w, new_b):
        pass
    def SGD_opitmiser(self):
        pass

    def MSE(y_hat, labels):
        pass

    def RMSE():
        pass

    def R2():
        pass

    
model = LinearRegression(n_features=8)
model