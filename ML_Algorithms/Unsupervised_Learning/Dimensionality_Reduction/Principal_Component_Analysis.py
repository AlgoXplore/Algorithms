import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LDA:
    """Fisher's Linear Discriminant Analysis (LDA) for binary classification."""
    def __init__(self):
        pass

    def fit(self, X, y):
        N = len(y)
        cls = np.unique(y)
        X1, X2 = X[y == cls[0]], X[y == cls[1]]
        p1, p2 = X1.shape[0] / N, X2.shape[0] / N
        mu1, mu2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
        cov1 = np.cov(X1, rowvar=False)
        cov2 = np.cov(X2, rowvar=False)
        cov_within = p1 * cov1 + p2 * cov2
        self.theta = np.linalg.inv(cov_within).dot(mu1 - mu2)
        self.b = -0.5 * (mu1 + mu2).dot(self.theta)
        self.cls = cls
        return self

    def predict(self, X):
        return np.where(X.dot(self.theta) + self.b > 0, self.cls[1], self.cls[0])


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Class {cl}', edgecolor='black')


def main():
    df = pd.read_csv('iris.data', header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # Extract setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0, 2]].values

    # Standardize features
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Train LDA
    lda = LDA()
    lda.fit(X_std, y)

    # Plot decision regions
    plt.figure()
    plot_decision_regions(X_std, y, classifier=lda)
    plt.title('Linear Discriminant Analysis (LDA)')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()