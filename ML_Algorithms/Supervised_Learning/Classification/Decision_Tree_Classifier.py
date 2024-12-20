import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Load and preprocess the penguins dataset
penguins = sns.load_dataset('penguins').dropna().reset_index(drop=True)
X = penguins.drop(columns='species')  # Features
y = penguins['species']              # Target variable

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

# Train Decision Tree Classifier
dtc = DecisionTreeClassifier(max_depth=10, min_samples_split=10)
dtc.fit(X_train, y_train)

# Evaluate model
y_test_hat = dtc.predict(X_test)
accuracy = np.mean(y_test_hat == y_test)
print(f"Test Accuracy: {accuracy:.4f}")