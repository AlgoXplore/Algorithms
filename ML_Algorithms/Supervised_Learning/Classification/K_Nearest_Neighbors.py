import numpy as np

def distance(x, y, p):
    # Compute the L_p distance between two points x and y.
    return sum(np.abs(x - y) ** p) ** (1 / p)

class KNN(object):
    def __init__(self, k=3, p=2):
        # Initialize the KNN classifier with k neighbors and L_p distance metric.
        self.k = k
        self.p = p

    def kdists(self, x_t):
        # Calculate distances from the test point x_t to all training points.
        all_dists = np.array([distance(x_t, x, self.p) for x in self.X])
        # Sort indices of training points by distance in ascending order.
        idx = sorted(range(len(all_dists)), key=lambda x: all_dists[x])
        # Return indices of the k nearest neighbors.
        return idx[:self.k]

    def vote(self, y):
        # Count occurrences of each class label among neighbors.
        v, c = np.unique(y, return_counts=True)
        # Return the label with the highest frequency.
        return v[np.argmax(c)]

    def fit(self, X, y):
        # Store training data and corresponding labels.
        self.X = X
        self.y = y

    def predict(self, X):
        # Predict labels for each point in the test dataset.
        return np.array([self.vote(self.y[self.kdists(x)]) for x in X])

def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from Perceptron import plot_decision_regions

    # Load the Iris dataset and extract petal length and width as features.
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # Petal length and petal width.
    y = iris.target  # Class labels.

    # Split the dataset into training and testing sets with stratified sampling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Create and train the KNN classifier.
    knn = KNN(k=5, p=2)
    knn.fit(X_train, y_train)

    # Plot decision regions using the trained KNN model.
    plt.figure()
    plot_decision_regions(X, y, classifier=knn)
    plt.title('KNN')
    plt.xlabel('Petal Length [cm]')
    plt.ylabel('Petal Width [cm]')
    plt.show()

if __name__ == '__main__':
    main()