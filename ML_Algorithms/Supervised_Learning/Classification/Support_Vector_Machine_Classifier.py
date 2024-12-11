import numpy as np

class SVMClassifier:
    def __init__(self, learning_rate=0.01, no_of_iterations=1000, lambda_parameter=0.01):
        # Initialize hyperparameters: learning rate, number of iterations, and regularization strength
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        # Map class labels to -1 and 1 for SVM computation
        Y = np.where(Y <= 0, -1, 1)
        self.m, self.n = X.shape  # Number of samples (m) and features (n)

        # Initialize weights and bias to zero
        self.w = np.zeros(self.n)
        self.b = 0

        # Perform gradient descent for the specified number of iterations
        for _ in range(self.no_of_iterations):
            self._update_weights(X, Y)

    def _update_weights(self, X, Y):
        for idx, x_i in enumerate(X):  # Iterate through each sample in the dataset
            # Check the condition for whether the sample lies outside the margin
            condition = Y[idx] * (np.dot(x_i, self.w) - self.b) >= 1

            if condition:
                # Update weights for correctly classified samples (no loss term)
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                # Update weights and bias for misclassified samples (include loss term)
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, Y[idx])
                db = -Y[idx]

            # Apply gradient descent to update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        # Compute the linear decision boundary
        linear_output = np.dot(X, self.w) - self.b
        # Determine class labels based on the sign of the linear output
        predicted_labels = np.sign(linear_output)
        # Map -1 back to 0 for binary classification
        return np.where(predicted_labels == -1, 0, 1)