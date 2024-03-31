# == Format tables to latex ==
# Read latest results.csv (from output directory)
# Format to the following latex table (as shown in the project description)
# \begin{table}[H]
# \begin{table}[ht]
# \centering
# \begin{tabular}{ccccc}
# \hline
# Outer fold & ANN & & Linear regression & baseline \\
# $i$ & $h^*_i$ & $E^{\text{test}}_i$ & $\lambda^*_i$ & $E^{\text{test}}_i$ & $E^{\text{test}}_i$ \\
# \hline
# 1 & 3 & 10.8 & 0.01 & 12.8 & 15.3 \\
# 2 & 4 & 10.1 & 0.01 & 12.4 & 15.1 \\
# $\vdots$ & $\vdots$ & $\vdots$ & $\vdots$ & $\vdots$ & $\vdots$ \\
# 10 & 3 & 10.9 & 0.05 & 12.1 & 15.9 \\
# \hline
# \end{tabular}
# \caption{Your table caption.}
# \label{your-table-label}
# \end{table}
import os
import pandas as pd 

output_dir = '../output'  
files = os.listdir(output_dir)  
files.sort(reverse=True)
latest_file = files[0]
results_df = pd.read_csv(f"{output_dir}/{latest_file}") 