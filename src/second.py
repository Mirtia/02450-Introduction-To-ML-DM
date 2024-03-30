import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sklearn.linear_model as lm
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn import linear_model, model_selection, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')

# read from dataset in .csv file
df=pd.read_csv("../data/SAheart.csv")

# display dataset's first rows, number of columns and rows and its attributes' types
print(df.head())
print(df.shape)
print(df.dtypes)

# drop the columns 'row.names' and 'adiposity'
df=df.drop('adiposity', axis=1)

# extract the information/datatype from the 'famhist' column and convert it to numeric
mapping = {'Present': 1, 'Absent': 0}
df['famhist'] = df['famhist'].map(mapping)
df['famhist'] = pd.to_numeric(df['famhist'])

# binarize typea with 55 as the threshold
df['typea'] = df['typea'].apply(lambda x: 1 if x >= 55 else 0)

# display none values
print(df.isna().sum())

# standardise data (subtract median and divide by deviation)
df['sbp'] = (df['sbp'] - df['sbp'].mean()) / df['sbp'].std()
df['tobacco'] = (df['tobacco'] - df['tobacco'].mean()) / df['tobacco'].std()
df['ldl'] = (df['ldl'] - df['ldl'].mean()) / df['ldl'].std()
df['obesity'] = (df['obesity'] - df['obesity'].mean()) / df['obesity'].std()
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['alcohol'] = (df['alcohol'] - df['alcohol'].mean()) / df['alcohol'].std()
df['famhist'] = (df['famhist'] - df['famhist'].mean()) / df['famhist'].std()
df['typea'] = (df['typea'] - df['typea'].mean()) / df['typea'].std()
df['chd'] = (df['chd'] - df['chd'].mean()) / df['chd'].std()

# table with statistics for the attributes 
print(df.describe())

X = df.drop('sbp', axis=1)
y = df['sbp'] # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define range of lambda values to test
lambdas = np.logspace(-2, 7, 50)

all_errors = {}

generalization_errors = []

# Define number of folds for KFold cross-validation
n_folds = 10

# Create KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_number = 0
fold_error_list = []
best_average_error = float('inf')  # Initialize with a large value
best_lambda = None
# Loop through each fold
for train_index, test_index in kf.split(X):
    fold_number += 1
    print(f"Fold {fold_number} of {n_folds}...")
    # Split data into training and testing sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize list to store error for each fold
    error_list = []

    for lmbda in lambdas:
        # Create Ridge regression object
        model = Ridge(lmbda)

        # Fit the model on the training data for this fold
        model.fit(X_train, y_train)
        
        # Make predictions on the testing data for this fold
        predictions = model.predict(X_test)
        
        # Calculate the mean squared error for this fold
        mse = mean_squared_error(y_test, predictions)
        
        # Append the error to the error list
        error_list.append(mse)

    fold_error_list.append(error_list)
    

    # Calculate the generalization error for this fold
    generalization_error = np.mean(error_list)

    err_dict = {'best_lambda': lambdas[error_list.index(min(error_list))], 'error': min(error_list)}

    all_errors[fold_number] = err_dict

    # Check if the current generalization error is smaller than the best found so far
    if generalization_error < best_average_error:
        best_average_error = min(error_list)
        best_lambda = lambdas[error_list.index(min(error_list))]  # Update best lambda

    # Append the generalization error to the list
    generalization_errors.append(generalization_error)

for error in all_errors:
    print(f"Fold {error}: {all_errors[error]}")

best_fold = 4

print(f"Best lambda: {best_lambda}")
print(f"Best generalization error: {best_average_error}")

#Average generalisation error vs regularisation parameter (lambda)
plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, fold_error_list[best_fold-1], marker='o')
plt.xlabel('λ (Regularization Parameter)')
plt.ylabel('Average Generalization Error')
plt.title('Generalization Error as a Function of λ')
plt.grid(True)
plt.show()

feature_names = df.drop(columns=['sbp']).columns
lambda_values = np.logspace(-2, 7, 50)
coefficients = []

#fitting Ridge regression models for different lambda values and storing coefficients
for alpha in lambda_values:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefficients.append(ridge.coef_)

#coefficient profile plot
plt.figure(figsize=(8, 6))
plt.plot(lambda_values, coefficients)
plt.xscale('log')
plt.xlabel('Lambda (Regularization Strength)')
plt.ylabel('Coefficients')
plt.title('Ridge Regression Coefficient Profile')
plt.legend(feature_names, loc='center left', bbox_to_anchor=(1, 0.5))

plt.axis('tight')
plt.show()

# Create Ridge regression object with the best lambda value
best_model = Ridge(alpha=best_lambda)

# Fit the model on the training data
best_model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = best_model.predict(X_test)

# 1. Coefficient Magnitudes Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, np.abs(best_model.coef_))
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Attribute')
plt.title('Magnitude of Coefficients')
plt.show()

# 2. Coefficient Significance Plot (using confidence intervals as error bars)
plt.figure(figsize=(10, 6))
ci = 1.96 * np.std(fold_error_list[best_fold-1]) / np.sqrt(len(fold_error_list[best_fold-1]))
plt.errorbar(range(len(feature_names)), best_model.coef_, yerr=ci, fmt='o', capsize=5)
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.xlabel('Attribute')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Significance with Confidence Intervals')
plt.grid(True)
plt.show()
