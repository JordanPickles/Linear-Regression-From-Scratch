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
    def __init__(self, optimiser, n_features):
        np.random.seed(10)
        self.w = np.random.randn(n_features,1)
        self.b = np.random.randn(1)
        self.optimiser = optimiser


    def preditctions(self, X):
        
        y_prediction = X @ self.w + self.b
        return y_prediction
        
    def fit_model(self, X, y):
        all_cost = []
        for epoch in range(self.optimiser.epochs) #TODO build the epochs into the 

            #Make predictions and update model        
            predictions = self.predict(X)
            new_w, new_b = self.optimiser.step(self.w, self.b, X, predictions, y) #TODO build this step in the optimiser class
            self._update_params(new_w, new_b)

            #Calculate the loss and append to a list to be visualised
            cost = self.MSE_loss()
            all_cost.append(cost)

        self.plot_cost(all_cost)
        print(f"Final Loss: {loss}")
        print(f"Weight values:  {self.w}")
        print(f"Bias values:    {self.b}")


    def _update_params(self, new_w, new_b):
        self.w = new_w # sets this instances weight to the new weight passed to the function
        self.b = new_b #sets this intances bias to the new bias value passed to the function
    

    def MSE_loss(y_hat, labels):
        errors = y_hat - labels
        squared_errors = errors ** 2
        mean_squared_error = sum(squared_errors) / len(squared_errors)
        return mean_squared_error
        

    def RMSE_loss(self):
        root_mean_squared_error = np.sqrt(self.MSE_loss())
        return root_mean_squared_error

    def R2(self):
        pass

    def plot_cost(self):
        pass

class SGDOptimiser():
    def __init__(self):


if __name__ == "__main__":
    optimiser = SGDOptimiser()
    model = LinearRegression(optimiser, n_features=8)
