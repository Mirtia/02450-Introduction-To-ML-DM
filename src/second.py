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

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# read from dataset in .csv file
df=pd.read_csv("../data/SAheart.csv")

# display dataset's first rows, number of columns and rows and its attributes' types
print(df.head())
print(df.shape)
print(df.dtypes)

# drop the columns 'row.names' and 'adiposity'
df=df.drop('adiposity', axis=1)
df=df.drop('id', axis=1)

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

    for lambda_ in lambdas:
        # Create Ridge regression object
        model = Ridge(lambda_)

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
# plt.figure(figsize=(10, 6))
# plt.semilogx(lambdas, fold_error_list[best_fold-1], marker='o')
# plt.xlabel('λ (Regularization Parameter)')
# plt.ylabel('Average Generalization Error')
# plt.title('Generalization Error as a Function of λ')
# plt.grid(True)
# plt.show()

feature_names = df.drop(columns=['sbp']).columns
lambda_values = np.logspace(-2, 7, 50)
coefficients = []

#fitting Ridge regression models for different lambda values and storing coefficients
for alpha in lambda_values:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefficients.append(ridge.coef_)

#coefficient profile plot
# plt.figure(figsize=(8, 6))
# plt.plot(lambda_values, coefficients)
# plt.xscale('log')
# plt.xlabel('Lambda (Regularization Strength)')
# plt.ylabel('Coefficients')
# plt.title('Ridge Regression Coefficient Profile')
# plt.legend(feature_names, loc='center left', bbox_to_anchor=(1, 0.5))

# plt.axis('tight')
# plt.show()

# Create Ridge regression object with the best lambda value
best_model = Ridge(alpha=best_lambda)

# Fit the model on the training data
best_model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = best_model.predict(X_test)

# 1. Coefficient Magnitudes Plot
# plt.figure(figsize=(10, 6))
# plt.barh(feature_names, np.abs(best_model.coef_))
# plt.xlabel('Coefficient Magnitude')
# plt.ylabel('Attribute')
# plt.title('Magnitude of Coefficients')
# plt.show()

# 2. Coefficient Significance Plot (using confidence intervals as error bars)
# plt.figure(figsize=(10, 6))
# ci = 1.96 * np.std(fold_error_list[best_fold-1]) / np.sqrt(len(fold_error_list[best_fold-1]))
# plt.errorbar(range(len(feature_names)), best_model.coef_, yerr=ci, fmt='o', capsize=5)
# plt.xticks(range(len(feature_names)), feature_names, rotation=45)
# plt.xlabel('Attribute')
# plt.ylabel('Coefficient Value')
# plt.title('Coefficient Significance with Confidence Intervals')
# plt.grid(True)
# plt.show()

# Regression part b

# Comparison of linear regression Ridge model with an artificial Neural Network and
# a baseline model

class BaselineModel:
    # As a baseline model, we will apply a linear regression model with no
    # features, i.e. it computes the mean of y on the training data, and use this value
    # to predict y on the test data
    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        return np.full(X.shape[0], self.mean)

# Define a neural network model
class NeuralNetwork(torch.nn.Module):
    # Experiment with the number of hidden units
    def __init__(self, n_features, n_hidden_units):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(n_features, n_hidden_units)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(n_hidden_units, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check if X_train, y_train, X_val, y_val are pandas DataFrames/Series and convert them to NumPy arrays
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_numpy()

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_tensor)
        mse = mean_squared_error(y_val, preds.numpy())
    return mse

# Test for different hidden unit sizes (1 to 10)
hidden_unit_sizes = list(range(1, 17))

# Two-thirds rule for hidden units
# (2/3) * 8  + 1 = 6.33 so we can try till 2 x (number of features) = 16

# Same lambda values as in the previous task (see lambdas)

# == Implement two-level cross-validation to compare the models with K1 = K2 = 10 ==

# 1. For ANN, test for different hidden unit sizes (range 1 to 16)
# 2. For Ridge regression, test for different lambda values (same as before)
# 3. For baseline model, no hyperparameters to tune
# Initialize KFold
outer_kfold = KFold(n_splits=10, shuffle=True, random_state=42)
inner_kfold = KFold(n_splits=10, shuffle=True, random_state=42)

results = []
# Outer loop
for i, (train_idx, test_idx) in enumerate(outer_kfold.split(X)):
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

    best_nn_h = None
    best_ridge_lambda = None
    lowest_nn_error = np.inf
    lowest_ridge_error = np.inf

    # Neural Network Inner Loop for Hyperparameter Tuning
    print("Outer Fold:", i + 1)
    for h in hidden_unit_sizes:
        print("Inner Fold:", i + 1, "Hidden Units:", h)
        nn_errors = []
        for inner_train_idx, inner_val_idx in inner_kfold.split(X_train_outer):
            # Splitting for inner loop
            X_train_inner, X_val_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_val_idx]
            y_train_inner, y_val_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_val_idx]
            
            # Train and evaluate the neural network model
            model = NeuralNetwork(n_features=X_train_inner.shape[1], n_hidden_units=h)
            mse = train_and_evaluate_model(model, X_train_inner, y_train_inner, X_val_inner, y_val_inner)
            nn_errors.append(mse)

        # Average error for this hidden unit size
        avg_mse = np.mean(nn_errors)
        if avg_mse < lowest_nn_error:
            lowest_nn_error = avg_mse
            best_nn_h = h

    # Ridge Regression Inner Loop for Hyperparameter Tuning
    for lambda_val in lambdas:
        print("Inner Fold:", i + 1, "Lambda:", lambda_val)
        ridge_errors = []
        for inner_train_idx, inner_val_idx in inner_kfold.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_val_idx]
            y_train_inner, y_val_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_val_idx]
            
            # Train and evaluate Ridge regression model
            model = Ridge(alpha=lambda_val)
            model.fit(X_train_inner, y_train_inner)
            y_pred = model.predict(X_val_inner)
            mse = mean_squared_error(y_val_inner, y_pred)
            ridge_errors.append(mse)

        # Average error for this lambda
        avg_mse = np.mean(ridge_errors)
        if avg_mse < lowest_ridge_error:
            lowest_ridge_error = avg_mse
            best_ridge_lambda = lambda_val

    # Baseline Model Error
    baseline_preds = np.full(shape=y_test_outer.shape, fill_value=y_train_outer.mean())
    baseline_error = mean_squared_error(y_test_outer, baseline_preds)

    # Store results
    results.append({
        'Fold': i + 1,
        'NN_h*': best_nn_h,
        'Ridge_λ*': best_ridge_lambda,
        'NN_Error': lowest_nn_error,
        'Ridge_Error': lowest_ridge_error,
        'Baseline_Error': baseline_error
    })

# Create and display the results DataFrame
results_df = pd.DataFrame(results)
print(results_df)