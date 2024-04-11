import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Read dataset from a CSV file
df = pd.read_csv("SAheart.csv")

# Drop columns 'id' and 'adiposity'
df = df.drop(['id', 'adiposity'], axis=1)

# Convert 'famhist' column to numeric
mapping = {'Present': 1, 'Absent': 0}
df['famhist'] = df['famhist'].map(mapping)
df['famhist'] = pd.to_numeric(df['famhist'])

# Binarize 'typea' with 55 as the threshold
df['typea'] = df['typea'].apply(lambda x: 1 if x >= 55 else 0)

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('chd', axis=1))
df_scaled = np.hstack((df_scaled, df['chd'].values.reshape(-1, 1)))
df = pd.DataFrame(df_scaled, columns=df.columns)

# Define features (X) and target (y)
X = df.drop('chd', axis=1)
y = df['chd']

# Define classifiers
classifiers = {
    "Baseline": DummyClassifier(strategy="most_frequent"),  # Baseline model predicts most frequent class
    "Logistic Regression": LogisticRegression(),  # Logistic Regression
    "ANN": MLPClassifier(),  # Artificial Neural Networks
    "CT": DecisionTreeClassifier(),  # Classification Trees
    "KNN": KNeighborsClassifier(),  # k-Nearest Neighbors
    "NB": GaussianNB()  # Naive Bayes
}

# Define parameters for cross-validation
K_outer = 10  # Number of outer folds for cross-validation

# Store test errors for each outer fold
test_errors = {clf_name: [] for clf_name in classifiers.keys()}

# Perform cross-validation
outer_cv = KFold(n_splits=K_outer, shuffle=True, random_state=42)
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Test each classifier on the outer fold and store test error
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_error = 1 - accuracy_score(y_test, y_pred)
        test_errors[clf_name].append(test_error)

# Compute average test errors across all outer folds for each classifier
avg_test_errors = {clf_name: np.mean(errors) for clf_name, errors in test_errors.items()}

# Output results
print("Average Test Errors:")
for clf_name, error in avg_test_errors.items():
    print(f"{clf_name}: {error:.4f}")
