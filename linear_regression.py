import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
from aicore.ml import data

# 15% for validation and test sets, 70% for training in total
X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5
)
 # Normalises the data
X_train, X_validation, X_test = data.standardize_multiple(X_train, X_validation, X_test)



class LinearRegression():
    def __init__(self, optimiser, n_features):
        """This is the constructor of the LinearRegression class.
        It takes two arguments as input:
            optimiser: an instance of a class that implements the optimizer's step method
            n_features: number of features in the input data
        It initializes the model's parameters (weights and bias) with random values, and assigns the optimizer to the optimizer attribute."""
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()
        self.optimiser = optimiser


    def _predictions(self, X): 
        """This method makes predictions based on the input features and the model's parameters.
        It takes one argument as input:
            X: input features
        It calculates the dot product of the input features and the model's weight values and adds the bias value.
        Returns:
            y_prediction: predicted output"""       
        y_prediction = X @ self.w + self.b
        
        return y_prediction
        
    def fit_model(self, X, y):
        """This the public method to be called to train the model.
        It takes two arguments as input:
            X: input features
            y: labels
        It iterates over the number of epochs specified by the optimizer's epochs attribute.
        In each iteration:
            1) It calls the _predictions method and passes the features as an argument
            2) It updates the model's parameters (self.w and self.b) using the optimizer's step method, which takes the current parameters, the input features, the predictions, and the labels as input.
            3) It calculates the mean squared error loss using the _MSE_loss method, which takes the predictions and the labels as input. It appends the loss to a list called all_cost.
        At the end, it plots the loss using the _plot_loss method and prints the final loss, weight values and bias values."""
       
        all_cost = []
        for epoch in range(self.optimiser.epochs): 
    
            predictions = self._predictions(X)
            new_w, new_b = self.optimiser.step(self.w, self.b, X, predictions, y) 
            self._update_params(new_w, new_b)

            cost = self._MSE_loss(predictions, y)
            all_cost.append(cost)

        self._plot_loss(all_cost)
        print(f"Final Loss: {cost}") 
        print(f"Weight values:  {self.w}")
        print(f"Bias values:    {self.b}")


    def _update_params(self, new_w, new_b):
        """Private method to update the current weights and biases with the new weights and biases following the optimisation process"""
        self.w = new_w 
        self.b = new_b 
    

    def _MSE_loss(self, y_hat, labels):
        """This method calculates the mean squared error loss between the predicted output and the true labels.
        It takes two arguments as input:
            y_hat: predicted output
            labels: true output
        It calculates the difference between the predicted output and the true labels, squares the difference,
        and takes the mean of the squared errors.
        Returns:
            mean_squared_error: the mean squared error loss"""
        errors = y_hat - labels
        squared_errors = errors ** 2
        mean_squared_error = sum(squared_errors) / len(squared_errors)
        return mean_squared_error
        

    def _plot_loss(self, losses):
        """Helper function for plotting the loss against the epoch"""
        plt.figure() 
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.plot(losses) 
        plt.show()

    

class SGDOptimiser():
    def __init__(self, lr, epochs):
        """
        This is the constructor of the SGDOptimizer class.
        It takes two arguments as input:
            lr: learning rate
            epochs: number of iterations over the training dataset
        It initializes the learning rate and number of epochs attributes"""
        self.lr = lr 
        self.epochs = epochs
    
    def _calculate_derivatives(self, features, predictions, labels): 
        """ This method calculates the gradients of the cost function with respect to the weights and biases.
        It takes three arguments as input: 
            features: input features on which predictions are made
            predictions: the predicted output
            labels: the true output
        Returns:
            dLdw: gradient of the cost function with respect to weights
            dLdb: gradient of the cost function with respect to bias"""
        m = len(labels)
        diffs = predictions - labels
        dLdw = 2* np.sum(features.T * diffs).T / m
        dLdb = 2 * np.sum(diffs) / m
        return dLdw, dLdb

    def step(self, w, b, features, predictions, labels):
        """This method updates the weights and biases using the gradients obtained from calculate_derivatives method
        It takes 5 arguments as input: 
            w: current weight values
            b: current bias value
            features: input features on which predictions are made
            predictions: predicted output
            labels: true output
            Returns:
            new_w : updated weight values
            new_b : updated bias value"""
        dLdw, dLdb = self._calculate_derivatives(features, predictions, labels)
        new_w = w - self.lr * dLdw
        new_b = b - self.lr * dLdb
        return new_w, new_b

if __name__ == "__main__":
    learning_rate = 0.001
    num_epochs = 1000
    optimiser = SGDOptimiser(lr = learning_rate, epochs = num_epochs)
    model = LinearRegression(optimiser = optimiser, n_features= 8)
    model.fit_model(X_train, y_train)
