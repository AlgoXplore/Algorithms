import numpy as np
import scipy.stats as ss

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        # Identify unique classes in the target labels
        self.cls = np.unique(y)
        
        # Get the number of samples and features
        N, self.N_feature = X.shape

        # Calculate the prior probabilities for each class
        self.priors = np.array([np.mean(y == cl) for cl in self.cls])

        # Compute the mean and standard deviation for each feature per class
        self.m = [np.mean(X[y == cl, :], axis=0) for cl in self.cls]
        self.std = [np.std(X[y == cl, :], axis=0) for cl in self.cls]

        return self

    def predict(self, X):
        # Predict the class for each sample in X
        return np.array([self.predict1(x) for x in X])

    def predict1(self, x):
        # Calculate likelihoods for all classes for a single sample
        likelihoods = []
        for mean, std, prior in zip(self.m, self.std, self.priors):
            # Compute the probability density for each feature
            prob = ss.norm(mean, std).pdf(x)

            # Combine probabilities with the prior probability
            likelihoods.append(np.prod(prob) * prior)

        # Return the class with the maximum likelihood
        return self.cls[np.argmax(likelihoods)]

def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions

    # Load the Iris dataset
    iris = datasets.load_iris()

    # Use only petal length and petal width features
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Initialize and train the Naive Bayes classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Visualize decision boundaries
    plt.figure()
    plot_decision_regions(X, y, clf=nb)
    plt.title('Naive Bayes')
    plt.xlabel('Petal Length [cm]')
    plt.ylabel('Petal Width [cm]')
    plt.show()

if __name__ == '__main__':
    main()
