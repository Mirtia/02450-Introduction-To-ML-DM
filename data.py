import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("projectdataset.csv")
df.head()
df.shape
df.dtypes

# drop the column 'row.names'
df=df.drop('row.names', axis=1) 
df=df.drop('adiposity', axis=1)

# extract the information/datatype from the 'famhist' column and convert it to numeric
mapping = {'Present': 1, 'Absent': 0}
df['famhist'] = df['famhist'].map(mapping)
df['famhist'] = pd.to_numeric(df['famhist'])

# binarize typea with 55 as the threshold
df['typea'] = df['typea'].apply(lambda x: 1 if x >= 55 else 0)

# table with statistics for the attributes 
print(df.describe())

# Pearson correlation heatmap, to investigate the connection of CHD with other attributes (BEFORE STANDARDIZATION):
df_cor=df[['chd','sbp','tobacco','ldl','famhist','typea','obesity','age','alcohol']]
num_columns = len(df_cor.columns)
plt.figure(figsize=(num_columns*1.5, num_columns))
sb.heatmap(df_cor.corr(method="pearson"), annot=True, annot_kws={"size":10})
plt.show()

# Boxplots for the continuous attributes
boxplot_columns = ['sbp', 'tobacco', 'ldl', 'obesity', 'alcohol', 'age']
boxplot_num = len(boxplot_columns)

plt.figure(figsize=(10, 6))
sb.boxplot(data=df[boxplot_columns], orient="h", showfliers=True, showmeans=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"),meanline = True, meanprops = dict(color = "green", linewidth=1), medianprops = dict(color = "blue", linewidth = 1))
plt.xlabel("Value")
plt.ylabel("Attribute")
plt.grid(True)
plt.show()

# Matrix of scatterplots of the attributes
column_scatter_1 = ['sbp', 'tobacco', 'ldl', 'famhist'] 
column_scatter_2 = ['typea', 'obesity', 'age', 'alcohol']
num_cols_1 = len(column_scatter_1)
num_cols_2 = len(column_scatter_2)


fig, axes = plt.subplots(num_cols_1, num_cols_1, figsize=(15, 15))

for i, column1 in enumerate(column_scatter_1):
    for j, column2 in enumerate(column_scatter_1):
        df.plot(kind='scatter', x=column1, y=column2, c='chd', cmap='coolwarm', alpha=0.7, ax=axes[i, j])
        axes[i, j].set_title(f"Scatterplot of {column1} with {column2}")

plt.tight_layout()

fig, axes = plt.subplots(num_cols_2, num_cols_2, figsize=(15, 15))

for i, column1 in enumerate(column_scatter_2):
    for j, column2 in enumerate(column_scatter_2):
        df.plot(kind='scatter', x=column1, y=column2, c='chd', cmap='coolwarm', alpha=0.7, ax=axes[i, j])
        axes[i, j].set_title(f"Scatterplot of {column1} with {column2}")

plt.tight_layout()

fig, axes = plt.subplots(num_cols_1, num_cols_2, figsize=(15, 15))

for i, column1 in enumerate(column_scatter_1):
    for j, column2 in enumerate(column_scatter_2):
        df.plot(kind='scatter', x=column1, y=column2, c='chd', cmap='coolwarm', alpha=0.7, ax=axes[i, j])
        axes[i, j].set_title(f"Scatterplot of {column1} with {column2}")

plt.tight_layout()

# Plot the distribution of the attributes (before standardization and left skewness normalization)
sb.set_style("whitegrid", {"grid_linestyle": "--"})
columns = ['sbp', 'tobacco', 'ldl', 'obesity', 'alcohol']
num_plots = len(columns)
num_rows = math.ceil(num_plots / 2)
num_columns = 2

fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))
axes = axes.ravel()
for i, column in enumerate(columns):
    sb.histplot(df[column], ax=axes[i], kde=True)
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)
    plt.xlim(xmin=0)

if num_plots % 2 != 0:
    axes.flat[-1].set_visible(False)

plt.show()

# standardise data (subtract median and divide by deviation)
df['sbp'] = (df['sbp'] - df['sbp'].mean()) / df['sbp'].std()
df['tobacco'] = (df['tobacco'] - df['tobacco'].mean()) / df['tobacco'].std()
df['ldl'] = (df['ldl'] - df['ldl'].mean()) / df['ldl'].std()
df['obesity'] = (df['obesity'] - df['obesity'].mean()) / df['obesity'].std()
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['alcohol'] = (df['alcohol'] - df['alcohol'].mean()) / df['alcohol'].std()
df['famhist'] = (df['famhist'] - df['famhist'].mean()) / df['famhist'].std()
df['typea'] = (df['typea'] - df['typea'].mean()) / df['typea'].std()

# apply left skewness normalization in 'tobacco' column
# transform the data using a logarithmic transformation
# This helps to reduce the left skewness and make the distribution more symmetric.
df['tobacco'] = np.where(df['tobacco'] > 0, np.log(df['tobacco']), df['tobacco'])
df['alcohol'] = np.where(df['alcohol'] > 0, np.log(df['alcohol']), df['alcohol'])

# table with statistics for the attributes 
print(df.describe())

# Plot the distribution of the attributes
sb.set_style("whitegrid", {"grid_linestyle": "--"})
columns = ['sbp', 'tobacco', 'ldl', 'obesity', 'alcohol']
num_plots = len(columns)
num_rows = math.ceil(num_plots / 2)
num_columns = 2

fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))
axes = axes.ravel()
for i, column in enumerate(columns):
    sb.histplot(df[column], ax=axes[i], kde=True)
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)
    plt.xlim(xmin=0)

if num_plots % 2 != 0:
    axes.flat[-1].set_visible(False)

plt.show()

# Attributes for QQ plot
qqplot_attributes1 = ['tobacco', 'alcohol']
qqplot_attributes2 = ['ldl', 'obesity']
qqplot_attributes3 = ['sbp']

# Create subplots
fig, axes = plt.subplots(len(qqplot_attributes1), 1, figsize=(8, 6*len(qqplot_attributes1)))

# Plot QQ plots for each attribute
for i, attr in enumerate(qqplot_attributes1):
    sm.qqplot(df[attr], line ='45', ax=axes[i])
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Create subplots
fig, axes = plt.subplots(len(qqplot_attributes2), 1, figsize=(8, 6*len(qqplot_attributes2)))

# Plot QQ plots for each attribute
for i, attr in enumerate(qqplot_attributes2):
    sm.qqplot(df[attr], line ='45', ax=axes[i])
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Create subplots
fig, axes = plt.subplots(len(qqplot_attributes3), 1, figsize=(8, 6*len(qqplot_attributes3)))

# Plot QQ plots for each attribute
sm.qqplot(df['sbp'], line ='45', ax=axes)
axes.grid(True)

plt.tight_layout()
plt.show()