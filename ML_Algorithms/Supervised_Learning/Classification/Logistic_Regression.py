import numpy as np

class Logistic_Regression:
    # Initialize the logistic regression model
    def __init__(self, learning_rate, no_of_iterations):
        """
        Constructor to initialize learning rate and number of iterations.
        :param learning_rate: The rate at which weights are updated during gradient descent.
        :param no_of_iterations: The number of iterations to run gradient descent.
        """
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        """
        Train the logistic regression model using gradient descent.
        :param X: Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.
        :param Y: Output vector of shape (m,), containing binary class labels (0 or 1).
        """
        self.m, self.n = X.shape  # Number of samples (m) and number of features (n)
        self.w = np.zeros(self.n)  # Initialize weights to zero
        self.b = 0  # Initialize bias to zero
        self.X = X  # Store input features
        self.Y = Y  # Store output labels
        
        # Perform gradient descent for the specified number of iterations
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        """
        Update the weights and bias using gradient descent.
        """
        # Compute predictions using the sigmoid function
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))
        
        # Compute gradients for weights and bias
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))  # Gradient of loss w.r.t weights
        db = (1 / self.m) * np.sum(Y_hat - self.Y)  # Gradient of loss w.r.t bias
        
        # Update weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict binary class labels for input data.
        :param X: Input feature matrix of shape (m, n) to predict labels for.
        :return: Binary class predictions (0 or 1) of shape (m,).
        """
        # Compute predictions using the sigmoid function
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        
        # Convert probabilities to binary class labels (threshold = 0.5)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred