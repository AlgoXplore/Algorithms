import numpy as np

class LDA(object):
    """The Fisher's Linear Discriminant Analysis classifier.
    Currently supports two-class problems only."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        N = len(y)
        cls = np.unique(y)
        X1, X2 = X[y == cls[0], :], X[y == cls[1], :]
        p1, p2 = X1.shape[0] / N, X2.shape[0] / N
        mu1, mu2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
        Q1 = np.dot((X1 - mu1).T, (X1 - mu1)) / (X1.shape[0] - 1)
        Q2 = np.dot((X2 - mu2).T, (X2 - mu2)) / (X2.shape[0] - 1)
        Q = p1 * Q1 + p2 * Q2
        self.theta = np.dot(np.linalg.inv(Q), mu1 - mu2)
        b1 = np.mean(np.dot(X1, self.theta))
        b2 = np.mean(np.dot(X2, self.theta))
        self.b = -0.5 * (b1 + b2)
        self.cls = cls
        return self

    def predict(self, X):
        return np.where(np.dot(X, self.theta) + self.b > 0, self.cls[1], self.cls[0])

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Helper function to visualize decision regions."""
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=[cmap(idx)],
            marker=markers[idx],
            label=f'Class {cl}',
        )

def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # Load Iris dataset
    iris = load_iris(as_frame=True)
    df = iris['data']
    df['species'] = iris['target_names'][iris['target']]

    # Extract setosa and versicolor
    df = df[df['species'].isin(['setosa', 'versicolor'])]
    y = np.where(df['species'] == 'setosa', 0, 1)
    X = df.iloc[:, [0, 2]].values

    # Standardize the features
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Instantiate and fit the LDA classifier
    lda = LDA()
    lda.fit(X_std, y)

    # Plot the decision regions
    plt.figure()
    plot_decision_regions(X_std, y, classifier=lda)
    plt.title('Fisher\'s Linear Discriminant Analysis')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()