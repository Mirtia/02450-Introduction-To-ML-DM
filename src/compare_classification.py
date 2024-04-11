import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kstest, lognorm, shapiro
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sklearn.linear_model as lm
from sklearn import tree, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import boxplot, figure, legend, plot, show, xlabel, ylabel
from sklearn.model_selection import KFold, RandomizedSearchCV

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

# Lambda values for Logistic Regression
lambda_values = np.logspace(-1, 1, 50)
C_values = 1 / lambda_values

# Define outer and inner CV splits
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=10, shuffle=True, random_state=43)

results = []

fold_number = 0
for train_idx, test_idx in outer_cv.split(X, y):
    fold_number += 1
    print(f"Fold {fold_number} running.")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Baseline model: predicting the most common class
    most_common_class = y_train.value_counts().idxmax()
    y_pred_base = [most_common_class] * len(y_test)
    error_rate_base = 1 - accuracy_score(y_test, y_pred_base)
    
    # Logistic Regression with CV for hyperparameter tuning
    best_acc_logreg = 0
    best_c = None
    for c in C_values:
        logreg = LogisticRegression(C=c, max_iter=1000, multi_class='multinomial', solver='lbfgs')
        inner_accs = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            X_inner_train, X_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
            y_inner_train, y_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
            
            logreg.fit(X_inner_train, y_inner_train)
            y_pred = logreg.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            inner_accs.append(acc)
        
        avg_acc = np.mean(inner_accs)
        if avg_acc > best_acc_logreg:
            best_acc_logreg = avg_acc
            best_c = c
    
    # ANN with Randomized Search for hyperparameter tuning
    ann = MLPClassifier(max_iter=1000, random_state=42)
    param_distributions = {
        'hidden_layer_sizes': [(1,), (5,), (10,), (15,) , (20,) , (25,) , (30,), (35,) , (40,), (45,), (50,) ]  # Testing different sizes for a single hidden layer
    }
    random_search = RandomizedSearchCV(ann, param_distributions, n_iter=4, cv=inner_cv, scoring='accuracy', random_state=44, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    best_ann = random_search.best_estimator_
    optimal_hidden_layer_size = best_ann.get_params()['hidden_layer_sizes']
    # Error rate for the best models on the test set
    logreg_best = LogisticRegression(C=best_c, max_iter=1000, multi_class='multinomial', solver='lbfgs')
    logreg_best.fit(X_train, y_train)
    y_pred_logreg = logreg_best.predict(X_test)
    
    y_pred_ann = best_ann.predict(X_test)
    
    error_rate_logreg = 1 - accuracy_score(y_test, y_pred_logreg)
    error_rate_ann = 1 - accuracy_score(y_test, y_pred_ann)
    
    results.append((error_rate_base, error_rate_logreg, error_rate_ann, best_c, optimal_hidden_layer_size))

for result in results:
    print(f"Baseline Error: {result[0]:.4f}, LogReg Error (C={1/result[3]}): {result[1]:.4f}, "
          f"ANN Error (hidden_layer_sizes={result[4]}): {result[2]:.4f}")
    



