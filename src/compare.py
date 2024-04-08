
# == Statistical comparison of the models ==
# 1. Ridge regression
# 2. ANN
# 3. Baseline Model
# Pairwise comparisons:
# - ANN vs. linear regression
# - ANN vs. baseline
# - linear regression vs. baseline
import pandas as pd
from scipy import stats
# We will use paired t-test
with open("../output/target.csv", 'r') as f:
    results_df = pd.read_csv(f) 

# ANN vs Ridge regression
t_statistic, p_value = stats.ttest_rel(results_df['NN_Error'], results_df['Ridge_Error'])
print(f"Ridge vs. ANN: t-statistic = {t_statistic}, p-value = {p_value}")

# ANN vs Baseline
t_statistic, p_value = stats.ttest_rel(results_df['Baseline_Error'], results_df['NN_Error'])    
print(f"Baseline vs. ANN: t-statistic = {t_statistic}, p-value = {p_value}")        


# Ridge Regression vs Baseline
t_statistic, p_value = stats.ttest_rel(results_df['Ridge_Error'], results_df['Baseline_Error'])     
print(f"Ridge vs. Baseline: t-statistic = {t_statistic}, p-value = {p_value}")  