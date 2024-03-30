import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import pca
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kstest, lognorm, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

# read from dataset in .csv file
df=pd.read_csv("projectdataset.csv")

# display dataset's first rows, number of columns and rows and its attributes' types
print(df.head())
print(df.shape)
print(df.dtypes)

# drop the columns 'row.names' and 'adiposity'
# for my dataset it was `id`
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

# table with statistics of the attributes 
print(df.describe())

# Statistical analysis - display of tests (KS test, Shapiro test, Log-normal test)
alpha = 0.05

columns = []
ks_results = []
shapiro_results = []
lognorm_results = []

# For storing p-values
ks_p_values = []
shapiro_p_values = []
lognorm_p_values = []

for column in df.columns:
    data_column = df[column]

    ks_statistic, ks_p_value = kstest(data_column, "norm")
    shapiro_statistic, shapiro_p_value = shapiro(data_column)
    lognorm_params = lognorm.fit(data_column)
    lognorm_statistic, lognorm_p_value = kstest(data_column, "lognorm", lognorm_params)
    
    columns.append(column)
    ks_results.append("Yes" if ks_p_value > alpha else "No")
    shapiro_results.append("Yes" if shapiro_p_value > alpha else "No")
    lognorm_results.append("Yes" if lognorm_p_value > alpha else "No")

    # Append p-values
    ks_p_values.append(ks_p_value)
    shapiro_p_values.append(shapiro_p_value)
    lognorm_p_values.append(lognorm_p_value)

results_df = pd.DataFrame({
    "Attribute": columns,
    "KS Test Normal": ks_results,
    "KS Test p-value": ks_p_values,
    "Shapiro Test Normal": shapiro_results,
    "Shapiro Test p-value": shapiro_p_values,
    "Log-Normal Test": lognorm_results,
    "Log-Normal Test p-value": lognorm_p_values
})

print(results_df)

# Analysis of feasibility for regression
correlation_matrix = df.corr()

# We have to device on the regression task. One idea would be to predict blood pressure

# Fit a linear regression model
X = df.drop('sbp', axis=1)  # independent variables
y = df['sbp']  # dependent variable

# Add a constant to the model (intercept)
X_const = sm.add_constant(X)

# OLS regression
model = sm.OLS(y, X_const).fit()

# Variance Inflation Factor (VIF) for checking multicollinearity
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(model.summary())

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
df['chd'] = (df['chd'] - df['chd'].mean()) / df['chd'].std()

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

# Drop chd attribute
df=df.drop('chd', axis=1)

# Number of PCA's and variance captured
# The number of components is experimental at this stage
# Choosing the correct number of principal components is crucial
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
variance_ratios = []

# Trying out the different numbers
for number in numbers:
  pca_ = PCA(n_components=number)
  pca_.fit_transform(df)
  variance_ratios.append(np.sum(pca_.explained_variance_ratio_))
  print(f"Number of components\t{number}\tTotal variance\t{sum(pca_.explained_variance_ratio_)}")

plt.figure(figsize=(2, 4))
plt.plot(numbers, variance_ratios, marker="o")
plt.xlabel("n_components")
plt.ylabel("Explained Variance Ratio")
plt.title("n_components vs. Explained Variance Ratio")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# PCA variance captured
model = pca.pca()
out = model.fit_transform(df)
model.plot()

plt.grid(linestyle="--", alpha=0.5, color="gray") 
plt.ylim(ymin=0)
plt.show()

# PCA variance ratios
explained_variance_ratios = pca_.explained_variance_ratio_
most_informative_pca = explained_variance_ratios.argmax()

# Print explained variance ratios for all PCs
for i, explained_variance_ratio in enumerate(explained_variance_ratios):
    print(f"PC_{i + 1}: {explained_variance_ratio:.4f}")

print(f"The most informative PCA is PCA_{most_informative_pca + 1} with an explained variance ratio of {explained_variance_ratios[most_informative_pca]:.4f}")

# Interpret PCA_1 and direction
factor_loadings_pc1 = pca_.components_[0]
feature_names = df.columns
feature_loadings = dict(zip(feature_names, factor_loadings_pc1))
sorted_features = sorted(feature_loadings.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the features with the highest absolute factor loadings for PC1
print("Features contributing most to PC1:")
for feature, loading in sorted_features:
    print(f"{feature}: {loading:.4f}")

# Try out 3, 4, 5 PCAs
number = 3
pca_ = PCA(n_components=number)
pca_.fit(df)
data_pca = pca_.transform(df)
data_pca = pd.DataFrame(data_pca, columns=["PCA_" + str(i) for i in range(number)])

# Correlation between PCAs
plt.figure(figsize=(12, 8))
factor_loadings = pca_.components_.T * np.sqrt(pca_.explained_variance_)
sb.heatmap(factor_loadings, cmap='coolwarm', annot=True, fmt='.2f')
plt.xlabel("Principal Components")
plt.ylabel("Variables")
plt.title("Factor Loadings Heatmap")
plt.show()

# Plotting Direction of Components
# Step 2: Perform PCA to reduce the data to 3 components
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df)

PCAs = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the projected points
ax.scatter(PCAs['PC1'], PCAs['PC2'], PCAs['PC3'], c='blue', marker='o', alpha=0.5)

for i in range(len(pca.components_)):
    vector = pca.components_[i] * max(PCAs.max())  # Scale vector for better visualization
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', label=f'PC{i+1}')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D PCA Plot with Loading Vectors')
plt.legend()
plt.show()

# Plotting projected columns to components
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df)

PCAs = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])

# Set up a 2x4 subplot (we have 8 attributes)
fig, axs = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, figsize=(20, 10))

axs = axs.flatten()

for i, attribute in enumerate(df.columns):
    ax = axs[i]
    
    attr_norm = (df[attribute] - df[attribute].min()) / (df[attribute].max() - df[attribute].min())
    
    img = ax.scatter(PCAs['PC1'], PCAs['PC2'], PCAs['PC3'], c=attr_norm, cmap='viridis')
    fig.colorbar(img, ax=ax, label=attribute, shrink=0.5, aspect=10)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'{attribute}')

#if len(df.columns) < 8:
#    axs[-1].set_visible(False)  # This hides the last subplot


plt.tight_layout()
plt.show()

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df)

PCAs = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])

fig, axs = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, figsize=(20, 10))

axs = axs.flatten()

# Arrow scale
arrow_scale = 4.0  # Adjust this value as needed

for i, attribute in enumerate(df.columns):
    ax = axs[i]
    
    # Normalize the attribute for coloring
    attr_norm = (df[attribute] - df[attribute].min()) / (df[attribute].max() - df[attribute].min())
    
    # Create a scatter plot
    img = ax.scatter(PCAs['PC1'], PCAs['PC2'], PCAs['PC3'], c=attr_norm, cmap='cividis')
    fig.colorbar(img, ax=ax, label=attribute, shrink=0.5, aspect=10)
    
    # Set labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'{attribute}')

    # Set the aspect of the plot to be equal
    ax.set_box_aspect([1,1,1])
    # Set plot limits
    plot_range = np.array([PCAs['PC1'].max() - PCAs['PC1'].min(), 
                           PCAs['PC2'].max() - PCAs['PC2'].min(), 
                           PCAs['PC3'].max() - PCAs['PC3'].min()]).max() / 2.0
    mid_x = (PCAs['PC1'].max() + PCAs['PC1'].min()) * 0.5
    mid_y = (PCAs['PC2'].max() + PCAs['PC2'].min()) * 0.5
    mid_z = (PCAs['PC3'].max() + PCAs['PC3'].min()) * 0.5
    
    ax.set_xlim(mid_x - plot_range, mid_x + plot_range)
    ax.set_ylim(mid_y - plot_range, mid_y + plot_range)
    ax.set_zlim(mid_z - plot_range, mid_z + plot_range)

    # Calculate the mean of the data for the origin of arrows
    mean_of_data = np.mean(principalComponents, axis=0)

    # Draw arrows
    for j in range(3):  # Assuming there are 3 principal components
        vector = pca.components_[j] * arrow_scale  # Use a fixed scaling factor
        ax.quiver(mean_of_data[0], mean_of_data[1], mean_of_data[2], 
                  vector[0], vector[1], vector[2], color=["r", "g", "b"][j], alpha=1.0, lw=2)

#if len(df.columns) < 10:
#    axs[-1].set_visible(False)

plt.tight_layout()
plt.show()