import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class RandomForest:
    def __init__(self):
        self.trees = []
        self.B = 0

    def fit(self, X_train, y_train, B=10, max_depth=100, seed=None):
        """
        Fit a Random Forest model with B decision trees.

        Parameters:
        - X_train: Training features (numpy array of shape (N, D))
        - y_train: Training labels (numpy array of shape (N,))
        - B: Number of trees
        - max_depth: Maximum depth of each tree
        - seed: Random seed for reproducibility
        """
        self.X_train = X_train
        self.N, self.D = X_train.shape
        self.y_train = y_train
        self.B = B
        self.seed = seed
        self.trees = []

        np.random.seed(seed)
        for b in range(self.B):
            # Bootstrap sampling
            sample = np.random.choice(np.arange(self.N), size=self.N, replace=True)
            X_train_b = X_train[sample]
            y_train_b = y_train[sample]

            # Fit a decision tree
            tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
            tree.fit(X_train_b, y_train_b)
            self.trees.append(tree)

    def predict(self, X_test):
        """
        Predict using the Random Forest model.

        Parameters:
        - X_test: Test features (numpy array of shape (M, D))

        Returns:
        - Predictions (numpy array of shape (M,))
        """
        # Collect predictions from all trees
        y_test_hats = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X_test)

        # Return the average prediction across all trees
        return y_test_hats.mean(axis=0)


# Example Dataset
np.random.seed(42)
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)
X_test = np.random.rand(20, 5)
y_test = np.random.rand(20)

# Build and Train the Random Forest Model
rf = RandomForest()
rf.fit(X_train, y_train, B=30, max_depth=20, seed=123)
y_test_hat = rf.predict(X_test)

# Plot Observed vs. Fitted Values
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_test_hat, ax=ax)
ax.set(
    xlabel=r'Observed Values ($y$)',
    ylabel=r'Predicted Values ($\hat{y}$)',
    title=r'Random Forest Observed vs. Fitted Values'
)
sns.despine()
plt.show()
