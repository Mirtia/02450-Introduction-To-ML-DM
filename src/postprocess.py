# == Comparison of the performance of the models ==
# \text{Average Test Error for ANN} = \frac{\sum_{i=1}^{10} E^{\text{test}}_i \text{(ANN)}}{10}
# and so on for the other models.

output_dir = '../output'  
files = os.listdir(output_dir)  
files.sort(reverse=True)
latest_file = files[0]
# Convert to latex table (with pandas)
results_df = pd.read_csv(f"{output_dir}/{latest_file}") 
# \text{Average Test Error for ANN} = \frac{\sum_{i=1}^{10} E^{\text{test}}_i \text{(ANN)}}{10}
# and so on for the other models.
# Compare the average test error of the models
average_test_error_ann = results_df['NN_Error'].mean()  
average_test_error_lin_reg = results_df['Ridge_Error'].mean()
average_test_error_baseline = results_df['Baseline_Error'].mean()
print(f"Average Test Error for ANN: {average_test_error_ann}")  
print(f"Average Test Error for Linear Regression: {average_test_error_lin_reg}")
print(f"Average Test Error for Baseline: {average_test_error_baseline}")
