#!/usr/bin/env python3
import os
import pandas as pd

# Base directory containing all method folders
base_dir = "/home/ok/mr30_ws/data/2204_sim/metric"

# Dictionary to store data from all CSVs
all_data = {}

# Method names and their parameters
methods = ["front", "grad_nbv", "grad_pso", "grad_sub", "grid", "rand", "samp"]
params = ["05", "10", "20", "30", "40", "50"]

# Read all CSV files
for method in methods:
    all_data[method] = {}
    for param in params:
        csv_path = os.path.join(base_dir, f"{method}{param}", "evaluation_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data[method][param] = df
            # Print structure of first CSV to understand its columns
            if method == methods[0] and param == params[0]:
                print(f"CSV structure:\n{df.head(3)}")

# List of metrics we want to include in the table
metrics_to_include = [
    "chamfer_distance",
    "hausdorff_distance",
    "f_score_5mm",
    "f_score_10mm", 
    "recall_5mm",
    "recall_10mm",
    "precision_5mm", 
    "precision_10mm"
]

# Helper function to escape underscores for LaTeX
def escape_for_latex(text):
    return text.replace("_", "\\_")

# Generate a LaTeX table
latex_output = []

# LaTeX table preamble
latex_output.append("\\begin{table}[htb]")
latex_output.append("\\centering")
latex_output.append("\\caption{Evaluation Metrics Comparison}")
latex_output.append("\\label{tab:evaluation_metrics}")

# Start table with better formatting
latex_output.append("\\begin{tabular}{l|l|" + "c"*len(params) + "}")
latex_output.append("\\hline")
latex_output.append("\\textbf{Method} & \\textbf{Metric} & " + " & ".join([f"\\textbf{{P{p}}}" for p in params]) + " \\\\")
latex_output.append("\\hline")

# For each method
for method in methods:
    # Skip if no data for this method
    if not all_data[method]:
        continue
        
    # For each metric
    for i, metric_name in enumerate(metrics_to_include):
        # Use escaped version of the metric name for display in the table
        escaped_metric_name = escape_for_latex(metric_name)
        row = [f"\\textbf{{{method}}}" if i == 0 else "", escaped_metric_name]
        
        # For each parameter value
        for param in params:
            if param in all_data[method]:
                # Find the row with this metric - use unescaped name for lookup
                metric_row = all_data[method][param][all_data[method][param]['Metric'] == metric_name]
                if not metric_row.empty:
                    value = metric_row['Value'].iloc[0]
                    # Format based on value magnitude
                    if abs(value) < 0.001:
                        row.append(f"{value:.2e}")
                    else:
                        row.append(f"{value:.4f}")
                else:
                    row.append("-")
            else:
                row.append("-")
                
        latex_output.append(" & ".join(row) + " \\\\")
    
    # Add horizontal line after each method
    latex_output.append("\\hline")

# Close table
latex_output.append("\\end{tabular}")
latex_output.append("\\end{table}")

# Write to file
with open("/home/ok/mr30_ws/evaluation_metrics_table.tex", "w") as f:
    f.write("\n".join(latex_output))

print(f"LaTeX table saved to /home/ok/mr30_ws/evaluation_metrics_table.tex")