#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# Base directory containing all method folders
base_dir = "/home/ok/mr30_ws/data/2504_realsense/metric"
log_dir = "/home/ok/mr30_ws/data/2504_realsense/log"
plots_dir = "/home/ok/mr30_ws/data/2504_realsense/evaluation_plots"
# Dictionary to store data from all CSVs
all_data = {}

# Method names and their parameters (steps)
methods = ["front", "grad_nbv", "grad_nbv_pso", "grad_nbv_sub", "grid", "rand", "samp"]
params = ["05", "10", "20", "25"]
# Convert params to numeric values for plotting
param_values = [int(p) for p in params]

# Method display names (nicer labels for plots)
method_display_names = {
    "front": "Frontier",
    "grad_nbv": "Gradient NBV",
    "grad_nbv_pso": "Gradient PSO",
    "grad_nbv_sub": "Gradient SA",
    "grid": "Grid",
    "rand": "Random",
    "samp": "Sampling"
}

# Colors for each method
colors = {
    "front": "#1f77b4",
    "grad_nbv": "#ff7f0e",
    "grad_nbv_pso": "#2ca02c",
    "grad_nbv_sub": "#d62728",
    "grid": "#9467bd",
    "rand": "#8c564b",
    "samp": "#e377c2"
}

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,         # Default font size
    'axes.titlesize': 17,     # Title font size
    'axes.labelsize': 17,     # Axis label font size
    #'xtick.labelsize': 12,    # X-axis tick label size
    #'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 17,    # Legend font size
    #'figure.titlesize': 18    # Figure title font size
})

# Helper functions for consistent text formatting across plots
def format_threshold(threshold):
    """Format threshold string for display (e.g., 2_5mm -> 2.5 mm)"""
    if threshold == "2_5mm":
        return "2.5 mm"
    elif threshold == "1mm":
        return "1 mm"
    elif threshold == "5mm":
        return "5 mm"
    elif threshold == "10mm":
        return "10 mm"
    else:
        return threshold

def format_metric_name(metric):
    """Format metric name for display (e.g., f_score_5mm -> F1 score (5 mm))"""
    # First handle special case for 2_5mm threshold
    if "_2_5mm" in metric:
        if metric.startswith("f_score"):
            return "F1 score (2.5 mm)"
        elif metric.startswith("precision"):
            return "Precision (2.5 mm)" 
        elif metric.startswith("recall"):
            return "Recall (2.5 mm)"
    
    if "_1mm" in metric:
        if metric.startswith("f_score"):
            return "F1 score (1 mm)"
        elif metric.startswith("precision"):
            return "Precision (1 mm)" 
        elif metric.startswith("recall"):
            return "Recall (1 mm)"
    # Normal case processing
    parts = metric.split('_')
    
    if parts[0] == "f":
        base_name = "F1 score"
    elif parts[0] == "precision":
        base_name = "Precision"
    elif parts[0] == "recall":
        base_name = "Recall"
    else:
        # For metrics like "chamfer_distance" or "hausdorff_distance"
        words = []
        for part in parts:
            words.append(part.capitalize())
        return " ".join(words)
    
    # Get threshold part
    if len(parts) > 1 and "mm" in parts[-1]:
        threshold = parts[-1]
        return f"{base_name} ({format_threshold(threshold)})"
    
    # Return just the base name for metrics without thresholds
    return base_name

# Metrics of interest
precision_recall_metrics = [
    "precision_1mm", "precision_2_5mm", "precision_5mm",
    "recall_1mm", "recall_2_5mm", "recall_5mm",
    "f_score_1mm", "f_score_2_5mm", "f_score_5mm"
]

distance_metrics = [
    "chamfer_distance", "hausdorff_distance"
]

# Read all CSV files
for method in methods:
    all_data[method] = {}
    for param in params:
        csv_path = os.path.join(base_dir, f"{method}{param}", "evaluation_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data[method][param] = df

# Create output directory for plots
os.makedirs(plots_dir, exist_ok=True)

# Helper function to get value for a specific metric
def get_metric_value(method, param, metric_name):
    if param not in all_data[method]:
        return None
    
    metric_row = all_data[method][param][all_data[method][param]['Metric'] == metric_name]
    if metric_row.empty:
        return None
    
    return metric_row['Value'].iloc[0]

# Helper function to create a dataframe for a specific metric across all methods and params
def create_metric_df(metric_name):
    data = []
    for method in methods:
        method_values = []
        for param in params:
            value = get_metric_value(method, param, metric_name)
            if value is not None:
                method_values.append(value)
            else:
                method_values.append(np.nan)
        data.append(method_values)
    
    df = pd.DataFrame(data, columns=param_values, index=methods)
    return df

#----------------------------------------
# 1. Line plots: Parameter vs. Metric for each method
#----------------------------------------
def generate_line_plots():
    # Plot precision and recall metrics
    for metric in precision_recall_metrics:
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            values = []
            valid_params = []
            
            for i, param in enumerate(params):
                value = get_metric_value(method, param, metric)
                if value is not None:
                    values.append(value)
                    valid_params.append(param_values[i])
            
            if values:  # Only plot if we have values
                plt.plot(valid_params, values, 'o-', label=method_display_names[method], 
                         color=colors[method], linewidth=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Number of Views')
        plt.ylabel(format_metric_name(metric))
        plt.title(f'{format_metric_name(metric)} by Number of Views')
        #plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'line_plot_{metric}.pdf'), bbox_inches='tight')
        plt.close()
    
    # Plot distance metrics
    for metric in distance_metrics:
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            values = []
            valid_params = []
            
            for i, param in enumerate(params):
                value = get_metric_value(method, param, metric)
                if value is not None:
                    values.append(value)
                    valid_params.append(param_values[i])
            
            if values:  # Only plot if we have values
                plt.plot(valid_params, values, 'o-', label=method_display_names[method], 
                         color=colors[method], linewidth=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Number of Views')
        plt.ylabel(format_metric_name(metric))
        plt.title(f'{format_metric_name(metric)} by Number of Views')
        
        # Use scientific notation for small values
        if metric == "chamfer_distance":
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            
        #plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'line_plot_{metric}.pdf'), bbox_inches='tight')
        plt.close()

#----------------------------------------
# 2. Bar plots: Comparing methods at specific parameter values
#----------------------------------------
def generate_bar_plots():
    # Generate bar plots for precision and recall at high parameter value (50 steps)
    high_param = params[-1]  # "50"
    
    for metric in precision_recall_metrics:
        plt.figure(figsize=(10, 6))
        
        values = []
        labels = []
        plot_colors = []
        
        for method in methods:
            value = get_metric_value(method, high_param, metric)
            if value is not None:
                values.append(value)
                labels.append(method_display_names[method])
                plot_colors.append(colors[method])
        
        if values:
            # Sort the values for better visualization
            sorted_indices = np.argsort(values)
            sorted_values = [values[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]
            sorted_colors = [plot_colors[i] for i in sorted_indices]
            
            plt.barh(sorted_labels, sorted_values, color=sorted_colors, height=0.6)
            plt.xlim(0, 1.05)  # Precision and recall are between 0 and 1
            plt.grid(True, axis='x', alpha=0.3)
            plt.xlabel(format_metric_name(metric))
            plt.title(f'{format_metric_name(metric)} Comparison ({high_param} Views)')
            
            # Add values on bars
            for i, v in enumerate(sorted_values):
                plt.text(v + 0.01, i, f'{v:.3f}', va='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'bar_plot_{metric}_{high_param}.pdf'), bbox_inches='tight')
            plt.close()
    
    # Generate bar plots for distance metrics at high parameter value
    for metric in distance_metrics:
        plt.figure(figsize=(10, 6))
        
        values = []
        labels = []
        plot_colors = []
        
        for method in methods:
            value = get_metric_value(method, high_param, metric)
            if value is not None:
                values.append(value)
                labels.append(method_display_names[method])
                plot_colors.append(colors[method])
        
        if values:
            # Sort the values (lower distances are better)
            sorted_indices = np.argsort(values)[::-1]  # Descending order
            sorted_values = [values[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]
            sorted_colors = [plot_colors[i] for i in sorted_indices]
            
            plt.barh(sorted_labels, sorted_values, color=sorted_colors, height=0.6)
            plt.grid(True, axis='x', alpha=0.3)
            plt.xlabel(format_metric_name(metric))
            plt.title(f'{format_metric_name(metric)} Comparison ({high_param} Views)')
            
            # Add values on bars
            for i, v in enumerate(sorted_values):
                if v < 0.001:
                    plt.text(v + 0.0001, i, f'{v:.2e}', va='center')
                else:
                    plt.text(v + 0.0001, i, f'{v:.4f}', va='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'bar_plot_{metric}_{high_param}.pdf'), bbox_inches='tight')
            plt.close()

#----------------------------------------
# 3. Heatmaps: All metrics for comparison
#----------------------------------------
def generate_heatmaps():
    # Create heatmap for precision and recall metrics at 50 views
    high_param = params[-1]  # "50"
    
    # Prepare data for heatmap
    methods_for_heatmap = []
    metrics_for_heatmap = []
    values_for_heatmap = []
    
    for method in methods:
        for metric in precision_recall_metrics + distance_metrics:
            value = get_metric_value(method, high_param, metric)
            if value is not None:
                methods_for_heatmap.append(method_display_names[method])
                metrics_for_heatmap.append(metric)
                values_for_heatmap.append(value)
    
    if values_for_heatmap:
        # Create DataFrame for heatmap
        heatmap_df = pd.DataFrame({
            'Method': methods_for_heatmap,
            'Metric': metrics_for_heatmap,
            'Value': values_for_heatmap
        })
        
        # Pivot the DataFrame
        pivot_df = heatmap_df.pivot(index='Method', columns='Metric', values='Value')
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        # Separate heatmaps for precision/recall and distance metrics
        # because of the different scales
        
        # 1. Precision and recall heatmap
        precision_recall_df = pivot_df[precision_recall_metrics]
        plt.figure(figsize=(12, 6))
        sns.heatmap(precision_recall_df, annot=True, cmap='viridis', fmt='.2f',
                    linewidths=.5, cbar_kws={'label': 'Value (higher is better)'})
        plt.title(f'Precision and Recall Metrics Comparison ({high_param} Views)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'heatmap_precision_recall_{high_param}.pdf'), bbox_inches='tight')
        plt.close()
        
        # 2. Distance metrics heatmap
        distance_df = pivot_df[distance_metrics]
        plt.figure(figsize=(8, 6))
        sns.heatmap(distance_df, annot=True, cmap='viridis_r', fmt='.4f',
                    linewidths=.5, cbar_kws={'label': 'Value (lower is better)'})
        plt.title(f'Distance Metrics Comparison ({high_param} Views)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'heatmap_distance_{high_param}.pdf'), bbox_inches='tight')
        plt.close()

#----------------------------------------
# 4. Progression plots: How metrics improve with more steps
#----------------------------------------
def generate_progression_plots():
    # Plot the improvement of F-score and Precision/Recall as number of views increases
    
    # F-score progression
    plt.figure(figsize=(12, 6))
    
    for method in methods:
        f_scores_1mm = []
        f_scores_2_5mm = []
        f_scores_5mm = []
        f_scores_10mm = []
        valid_params = []
        
        for i, param in enumerate(params):
            f1 = get_metric_value(method, param, "f_score_1mm")
            f2_5 = get_metric_value(method, param, "f_score_2_5mm")
            f5 = get_metric_value(method, param, "f_score_5mm")
            f10 = get_metric_value(method, param, "f_score_10mm")
            
            if f1 is not None and f2_5 is not None and f5 is not None and f10 is not None:
                f_scores_1mm.append(f1)
                f_scores_2_5mm.append(f2_5)
                f_scores_5mm.append(f5)
                f_scores_10mm.append(f10)
                valid_params.append(param_values[i])
        
        if f_scores_5mm and f_scores_10mm:
            plt.plot(valid_params, f_scores_1mm, '^-.', label=f"{method_display_names[method]} (1mm)", 
                     color=colors[method], linewidth=4, alpha=0.6)
            plt.plot(valid_params, f_scores_2_5mm, 'x-.', label=f"{method_display_names[method]} (2.5mm)", 
                     color=colors[method], linewidth=4, alpha=0.7)
            plt.plot(valid_params, f_scores_5mm, 'o-', label=f"{method_display_names[method]} (5mm)", 
                     color=colors[method], linewidth=4)
            plt.plot(valid_params, f_scores_10mm, 's--', label=f"{method_display_names[method]} (10mm)", 
                     color=colors[method], alpha=0.7, linewidth=4)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Views')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Progression with Increasing Views')
    #plt.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'progression_f_score.pdf'), bbox_inches='tight')
    plt.close()
    
    # Distance metrics progression
    plt.figure(figsize=(12, 6))
    
    for method in methods:
        chamfer_values = []
        valid_params = []
        
        for i, param in enumerate(params):
            chamfer = get_metric_value(method, param, "chamfer_distance")
            
            if chamfer is not None:
                chamfer_values.append(chamfer)
                valid_params.append(param_values[i])
        
        if chamfer_values:
            plt.plot(valid_params, chamfer_values, 'o-', label=method_display_names[method], 
                     color=colors[method], linewidth=4)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Views')
    plt.ylabel('Chamfer Distance')
    plt.title('Chamfer Distance Reduction with Increasing Views')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    #plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'progression_chamfer_distance.pdf'), bbox_inches='tight')
    plt.close()
#----------------------------------------
# 5. Time progression plots: How metrics improve over time
#----------------------------------------
def generate_time_progression_plots():
    """Generate plots showing metrics progression against time."""
    print("Generating time progression plots...")
    
    # Dictionary to store time data for each method
    time_data = {}
    
    # Read log files for each method
    for method in methods:
        log_file = os.path.join(log_dir, f"{method}.csv")
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                
                # Check which column format is used for time
                time_col = None
                if "total time" in df.columns:
                    time_col = "total time"
                elif "total_time" in df.columns:
                    time_col = "total_time"
                    
                if time_col is not None and "step" in df.columns:
                    # Map steps to corresponding times
                    time_data[method] = df.set_index("step")[time_col].to_dict()
                    print(f"  Loaded time data for {method} using column '{time_col}'")
                else:
                    print(f"  Missing required columns in {log_file}")
                    
            except Exception as e:
                print(f"  Error reading {log_file}: {e}")
    
    # Create plots for all metric types and thresholds
    thresholds = ["1mm", "2_5mm", "5mm", "10mm"]
    
    # Generate plots for each metric type (precision, recall, F-score)
    for metric_type in ["precision", "recall", "f_score"]:
        print(f"  Generating {metric_type} vs. time plots...")
        
        for threshold in thresholds:
            metric = f"{metric_type}_{threshold}"
            plt.figure(figsize=(12, 6))
            
            for method in methods:
                if method not in time_data:
                    continue
                    
                metric_values = []
                time_values = []
                
                for param in params:
                    step = int(param)  # Convert "05" to 5, "10" to 10, etc.
                    if step in time_data[method]:
                        value = get_metric_value(method, param, metric)
                        time_seconds = time_data[method][step]
                        
                        if value is not None:
                            metric_values.append(value)
                            time_values.append(time_seconds)
                
                if metric_values and time_values:
                    plt.plot(time_values, metric_values, 'o-', 
                             label=method_display_names[method],
                             color=colors[method], linewidth=4)
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time (seconds)')
            plt.ylabel(format_metric_name(metric))
            plt.title(f'{format_metric_name(metric)} Over Time')
            #plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'time_progression_{metric}.pdf'), bbox_inches='tight')
            plt.close()
    
    # Create plots for distance metrics vs time
    for metric in distance_metrics:
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            if method not in time_data:
                continue
                
            metric_values = []
            time_values = []
            
            for param in params:
                step = int(param)
                if step in time_data[method]:
                    value = get_metric_value(method, param, metric)
                    time_seconds = time_data[method][step]
                    
                    if value is not None:
                        metric_values.append(value)
                        time_values.append(time_seconds)
            
            if metric_values and time_values:
                plt.plot(time_values, metric_values, 'o-', 
                         label=method_display_names[method],
                         color=colors[method], linewidth=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (seconds)')
        plt.ylabel(format_metric_name(metric))
        plt.title(f'{format_metric_name(metric)} Over Time')
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        #plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'time_progression_{metric}.pdf'), bbox_inches='tight')
        plt.close()

#----------------------------------------
# 6. Distance traveled vs. Metric plots: Quality improvement vs. movement
#----------------------------------------
def generate_distance_traveled_plots():
    """Generate plots showing metrics improvement vs distance traveled by the camera."""
    print("Generating distance traveled vs. metric plots...")
    
    # Dictionary to store camera trajectory data for each method
    camera_data = {}
    
    # Read camera position files
    for method in methods:
        log_file = os.path.join(log_dir, f"{method}.csv")
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                # Check if we have the required columns
                if all(col in df.columns for col in ['camera_x', 'camera_y', 'camera_z', 'step']):
                    camera_data[method] = df
                    print(f"  Loaded camera trajectory for {method} ({len(df)} points)")
                else:
                    print(f"  Missing position columns in {log_file}")
            except Exception as e:
                print(f"  Error reading {log_file}: {e}")
    
    # Calculate cumulative distance for each method
    distance_data = {}
    for method in camera_data:
        df = camera_data[method]
        # Sort by step to ensure correct order
        df = df.sort_values('step')
        
        # Calculate distances between consecutive points
        distances = [0]  # First point has zero distance
        positions = df[['camera_x', 'camera_y', 'camera_z']].values
        
        for i in range(1, len(positions)):
            # Euclidean distance between current and previous position
            dist = np.linalg.norm(positions[i] - positions[i-1])
            distances.append(distances[-1] + dist)  # Add to cumulative distance
        
        # Map steps to distances
        step_to_distance = {}
        for step, dist in zip(df['step'], distances):
            step_to_distance[step] = dist
        
        distance_data[method] = step_to_distance
        print(f"  Total distance traveled for {method}: {distances[-1]:.3f} units")
    
    # Create plots for each metric type and threshold
    thresholds = ["1mm", "2_5mm", "5mm", "10mm"]
    
    # Generate F-score vs distance plots (keep the existing implementation)
    print("  Generating F-score vs. distance plots...")
    for threshold in thresholds:
        metric = f"f_score_{threshold}"
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            if method not in distance_data:
                continue
                
            metric_values = []
            distance_values = []
            
            for param in params:
                step = int(param)
                # Check if we have both distance data and metric value
                if step in distance_data[method]:
                    value = get_metric_value(method, param, metric)
                    dist = distance_data[method][step]
                    
                    if value is not None:
                        metric_values.append(value)
                        distance_values.append(dist)
            
            if metric_values and distance_values:
                plt.plot(distance_values, metric_values, 'o-', 
                         label=method_display_names[method],
                         color=colors[method], linewidth=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Cumulative Distance Traveled (m)')
        plt.ylabel(format_metric_name(metric))
        plt.title(f'{format_metric_name(metric)} by Camera Distance Traveled')
        #plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'distance_traveled_f_score_{threshold}.pdf'), bbox_inches='tight')
        plt.close()
    
    # Generate Precision vs distance plots
    print("  Generating Precision vs. distance plots...")
    for threshold in thresholds:
        metric = f"precision_{threshold}"
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            if method not in distance_data:
                continue
                
            metric_values = []
            distance_values = []
            
            for param in params:
                step = int(param)
                if step in distance_data[method]:
                    value = get_metric_value(method, param, metric)
                    dist = distance_data[method][step]
                    
                    if value is not None:
                        metric_values.append(value)
                        distance_values.append(dist)
            
            if metric_values and distance_values:
                plt.plot(distance_values, metric_values, 'o-', 
                         label=method_display_names[method],
                         color=colors[method], linewidth=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Cumulative Distance Traveled (m)')
        plt.ylabel(format_metric_name(metric))
        plt.title(f'{format_metric_name(metric)} by Camera Distance Traveled')
        #plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'distance_traveled_precision_{threshold}.pdf'), bbox_inches='tight')
        plt.close()
    
    # Generate Recall vs distance plots
    print("  Generating Recall vs. distance plots...")
    for threshold in thresholds:
        metric = f"recall_{threshold}"
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            if method not in distance_data:
                continue
                
            metric_values = []
            distance_values = []
            
            for param in params:
                step = int(param)
                if step in distance_data[method]:
                    value = get_metric_value(method, param, metric)
                    dist = distance_data[method][step]
                    
                    if value is not None:
                        metric_values.append(value)
                        distance_values.append(dist)
            
            if metric_values and distance_values:
                plt.plot(distance_values, metric_values, 'o-', 
                         label=method_display_names[method],
                         color=colors[method], linewidth=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Cumulative Distance Traveled (m)')
        plt.ylabel(format_metric_name(metric))
        plt.title(f'{format_metric_name(metric)} by Camera Distance Traveled')
        #plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'distance_traveled_recall_{threshold}.pdf'), bbox_inches='tight')
        plt.close()

def generate_standalone_legend():
    """Generate a standalone horizontal legend image with all method names and colors."""
    print("Generating standalone legend...")
    
    # Create a figure for the legend
    plt.figure(figsize=(10, 1))  # Width is larger than height for horizontal layout
    
    # Create dummy lines with the correct colors and labels but don't actually plot them
    lines = []
    for method in methods:
        line, = plt.plot([0], [0], '-', color=colors[method], label=method_display_names[method], linewidth=4)
        lines.append(line)
        
    # Create the legend
    legend = plt.legend(handles=lines, loc='center', ncol=len(methods), 
                        frameon=True, framealpha=1, facecolor='white')
    
    # Get the legend as a separate artist
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    # Hide the axis and frame
    plt.axis('off')
    
    # Save the legend as a PNG file
    plt.savefig(os.path.join(plots_dir, 'methods_legend.png'), 
                bbox_inches='tight', dpi=300, transparent=False)
    plt.close()
    
    print("  Standalone legend saved as methods_legend.png")

# Update the main execution block to include the new function
if __name__ == "__main__":
    print("Generating line plots...")
    #generate_line_plots()
    
    print("Generating bar plots...")
    #generate_bar_plots()
    
    print("Generating heatmaps...")
    generate_heatmaps()
    
    print("Generating progression plots...")
    #generate_progression_plots()
    
    # Fix for time progression plots
    print("Generating time progression plots...")
    #generate_time_progression_plots()
        
    print("Generating distance traveled vs. F-score plots...")
    #generate_distance_traveled_plots()
    
    #generate_standalone_legend()
    print(f"All plots saved to: {plots_dir}")

