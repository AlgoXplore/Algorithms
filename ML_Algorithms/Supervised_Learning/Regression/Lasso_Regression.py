import numpy as np

class Lasso_Regression():

    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        # Initialize hyperparameters
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        # Initialize weights and bias
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Perform gradient descent for a fixed number of iterations
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        # Predict target values
        Y_prediction = self.predict(self.X)
        dw = np.zeros(self.n)

        # Compute gradients for weights
        for i in range(self.n):
            if self.w[i] > 0:
                dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.m
            else:
                dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - Y_prediction)) - self.lambda_parameter) / self.m

        # Compute gradient for bias
        db = -2 * np.sum(self.Y - Y_prediction) / self.m

        # Update weights and bias using gradients
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        # Return predicted values
        return X.dot(self.w) + self.b