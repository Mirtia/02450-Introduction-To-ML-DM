import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# Read dataset from a CSV file
df=pd.read_csv("projectdataset.csv")

# drop the columns 'row.names' and 'adiposity'
df=df.drop('row.names', axis=1) 
df=df.drop('adiposity', axis=1)

# Convert 'famhist' column to numeric
mapping = {'Present': 1, 'Absent': 0}
df['famhist'] = df['famhist'].map(mapping)
df['famhist'] = pd.to_numeric(df['famhist'])

# Binarize 'typea' with 55 as the threshold
df['typea'] = df['typea'].apply(lambda x: 1 if x >= 55 else 0)

# Standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('chd', axis=1))
df_scaled = np.hstack((df_scaled, df['chd'].values.reshape(-1, 1)))
df = pd.DataFrame(df_scaled, columns=df.columns)

# Define features (X) and target (y)
X = df.drop('chd', axis=1)
y = df['chd']

# Define best lambda obtained
best_lambda = 9.102981779915218
best_c = 1 / best_lambda

# Fit logistic regression model with the best lambda on the entire dataset
logreg_best = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
logreg_best.fit(X, y)

# Output the final model
print("Final Logistic Regression Model:")
print(logreg_best)

# Print the coefficients
print("Coefficients:")
for i, feature in enumerate(X.columns):
    print(f"{feature}: {logreg_best.coef_[0][i]}")

# Generate Confusion Matrix
y_pred = logreg_best.predict(X)
cm = confusion_matrix(y, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(ticks=[0.5, 1.5], labels=['No CHD', 'CHD'])
plt.yticks(ticks=[0.5, 1.5], labels=['No CHD', 'CHD'])
plt.show()

# ROC Curve
y_prob = logreg_best.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 1. Coefficient Magnitudes Plot
feature_names = df.drop(columns=['chd']).columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, np.abs(logreg_best.coef_[0]))
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Attribute')
plt.title('Magnitude of Coefficients')
plt.show()