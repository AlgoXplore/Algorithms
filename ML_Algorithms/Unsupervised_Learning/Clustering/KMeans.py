import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random data
X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1

# Plot the data
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')  # Corrected the quotes
plt.show()

# Perform KMeans clustering
Kmean = KMeans(n_clusters=2, random_state=42)  # Added random_state for reproducibility
Kmean.fit(X)

# Display cluster centers
print("Cluster Centers:\n", Kmean.cluster_centers_)

# Plot data points and cluster centers
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.scatter(Kmean.cluster_centers_[0, 0], Kmean.cluster_centers_[0, 1], s=200, c='g', marker='s')  # Corrected the quotes
plt.scatter(Kmean.cluster_centers_[1, 0], Kmean.cluster_centers_[1, 1], s=200, c='r', marker='s')  # Corrected the quotes
plt.show()

# Display cluster labels
print("Cluster Labels:\n", Kmean.labels_)

# Test with a new sample
sample_test = np.array([-3.0, -3.0])
second_test = sample_test.reshape(1, -1)
prediction = Kmean.predict(second_test)
print("Prediction for the test point:", prediction)
