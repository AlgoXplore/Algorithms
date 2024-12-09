import numpy as np

class Linear_Regression:

    def __init__(self, learning_rate, no_of_iterations):
        """
        Constructor to initialize learning rate and number of iterations.
        :param learning_rate: The rate at which weights are updated during gradient descent.
        :param no_of_iterations: The number of iterations to run gradient descent.
        """
        # Initialize learning rate and number of iterations
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        """
        Train the linear regression model using gradient descent.
        :param X: Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.
        :param Y: Output vector of shape (m,), containing target values.
        """
        # Number of examples (m) and features (n)
        self.m, self.n = X.shape

        # Initialize weights and bias to zero
        self.w = np.zeros(self.n)  # Weights for the features
        self.b = 0  # Bias term
        self.X = X  # Store input features
        self.Y = Y  # Store output values

        # Perform Gradient Descent for the specified number of iterations
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        """
        Update weights and bias using gradient descent.
        """
        # Predicted output for the current weights and bias
        Y_prediction = self.predict(self.X)

        # Calculate gradients for weights (dw) and bias (db)
        dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m  # Gradient of loss w.r.t weights
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m  # Gradient of loss w.r.t bias

        # Update weights and bias using the gradients
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict output values for the input feature matrix.
        :param X: Input feature matrix of shape (m, n).
        :return: Predicted output vector of shape (m,).
        """
        # Calculate the linear model: Y = X.w + b
        return X.dot(self.w) + self.b