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

#import torch
import warnings

warnings.filterwarnings('ignore')

# read from dataset in .csv file
df=pd.read_csv("projectdataset.csv")

# display dataset's first rows, number of columns and rows and its attributes' types
print(df.head())
print(df.shape)
print(df.dtypes)

# drop the columns 'row.names' and 'adiposity'
df=df.drop('row.names', axis=1) 
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
lambdas = np.power(10.0, range(-10, 9))

all_errors = {}

generalization_errors = []

# Define number of folds for KFold cross-validation
n_folds = 10

# Create KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_number = 0
fold_error_list = []
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


    # Append the generalization error to the list
    generalization_errors.append(generalization_error)

for error in all_errors:
    print(f"Fold {error}: {all_errors[error]}")

# Calculate the average MSE for each lambda across all folds
average_errors = {}
for lmbda in lambdas:
    # Get the errors for all folds corresponding to this lambda
    errors_for_lambda = [all_errors[fold]['error'] for fold in all_errors if all_errors[fold]['best_lambda'] == lmbda]
    # Calculate the average error across all folds for this lambda
    average_errors[lmbda] = np.mean(errors_for_lambda)

# Choose the lambda with the lowest average MSE
best_lambda = min(average_errors, key=average_errors.get)
best_average_error = average_errors[best_lambda]

print(f"Best Lambda: {best_lambda}")
print(f"Average MSE with Best Lambda: {best_average_error}")

best_fold = 4

print(f"Minimum Generalization Error: {min(generalization_errors)}")
print(f"Lambda value with minimum generalization error: {lambdas[generalization_errors.index(min(generalization_errors))]}")

#Average generalisation error vs regularisation parameter (lambda)
plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, fold_error_list[best_fold-1], marker='o')
plt.xlabel('λ (Regularization Parameter)')
plt.ylabel('Average Generalization Error')
plt.title('Generalization Error as a Function of λ')
plt.grid(True)
plt.show()

# The generalization error curve remains low for a range of small lambda values (from 10^-10 to 10^2), being 10^2 the value of lambda where
# the average generalization error is minimumm, and therefore it has been selected has the lamnda value to be used for our linear regression
# model. From the fact that the plot exhibits a curve that remains low for a range of lambda values, we can conclude that the model benefits
# from regularization within that range of values, which helps to prevent overfitting by penalizing large coefficients, leading to a better
# generalization performance on unseen data. Regarding the shape of the curve, the lack of a "U-shapped" pattern suggests the model is
# relatively robust to overfitting across, which means that even when lambda increases beyond the optimal range, the model's performance does
# not deteriorate significantly. 

feature_names = df.columns
lambda_values = np.logspace(-1, 6, 1000)
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

# The ridge regression "penalizes" the variable coefficients, which means that the ones that are less effective to predict the target variable
# are faster to reach zero. In general, as alpha increases, the penalty for large coefficients also increases, decreasing the values of the 
# coefficients. Therefore, to analyse the plot correctly, it is important to notice the flattening process of the coefficients' curves that
# represent reaching the regularization state. If the coefficent's values remain large, then it may be a significant feature when
# making predictions of the "sbl" variable. However, if the attribute's cofficients curve gets flat earlier than others, that can mean the
# feature is insignificant when making predictions.

# In our case, the values of the coefficients seem to start all with values smaller than 0.30 and then flattens out as the lambda increases to
# 10^6. These results shows the effect of regularization, the importance of certain features, like the "alcohol" and "typea" attributes, 
# due to their stronger resistance to regularization and the seemingly lack of importance of other features, like the "ldl" and "famhist" attributes
# as they are the faster to reach to zero while increasing the value of lambda. Overall, the observed pattern in the ridge regression coefficients 
# reflects the interplay between regularization strength, feature importance, and model complexity, leading to improved generalization performance 
# and interpretability.