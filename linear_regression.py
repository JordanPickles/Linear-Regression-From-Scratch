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
    def __init__(self, n_features, optimiser):
        np.random.seed(10)
        self.w = np.random.randn(n_features,1)
        self.b = np.random.randn(1)
        self.optimiser = optimiser


    def preditctions(self, X):
        all_loss = []
        y_prediction = X @ self.w + self.b
        return y_prediction
        
    def fit_model(self, X, y):
        for epoch in range(self.optimiser.epochs) #TODO build the epochs into the 

            #Make predictions and update model        
            predictions = self.predict(X)
            new_w, new_b = self.optimiser.step(self.w, self.b, X, predictions, y) #TODO build this step in the optimiser class
            self._update_params(new_w, new_b)

            #Calculate the loss and append to a list to be visualised
            loss = self.MSE_loss()
            all_loss.append(loss)

        plot_loss(all_loss)
        print(f"Final Loss: {loss}")
        print(f"Weight values:  {self.w}")
        print(f"Bias values:    {self.b}")


        pass

    def _update_params(self, new_w, new_b):
        self.w = new_w # sets this instances weight to the new weight passed to the function
        self.b = new_b #sets this intances bias to the new bias value passed to the function
        pass

    def MSE_loss(y_hat, labels):
        pass

    def RMSE_loss():
        pass

    def R2():
        pass

class SGDOptimiser():
    def __init__(self):

    
model = LinearRegression(n_features=8)
optimiser = SGDOptimiser()