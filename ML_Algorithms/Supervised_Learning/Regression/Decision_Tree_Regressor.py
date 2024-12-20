import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load and preprocess the dataset
tips = sns.load_dataset('tips')
X = tips.drop(columns='tip')  # Features
y = tips['tip']              # Target variable

# Train-test split
np.random.seed(1)
test_frac = 0.25
test_size = int(len(y) * test_frac)
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace=False)
X_train = X.iloc[~np.isin(np.arange(len(y)), test_idxs)]
y_train = y.iloc[~np.isin(np.arange(len(y)), test_idxs)]
X_test = X.iloc[test_idxs]
y_test = y.iloc[test_idxs]

# Encode categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Train the Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth=7, min_samples_split=5)
dtr.fit(X_train, y_train)

# Make predictions on the test set
y_test_hat = dtr.predict(X_test)

# Visualize actual vs predicted values
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_test_hat, ax=ax)
ax.set(xlabel=r'$y$', ylabel=r'$\hat{y}$', title=r'Test Sample $y$ vs. $\hat{y}$')
sns.despine()
plt.show()