import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path


def find_metrics_files(base_dir):
    """
    Find all metrics.csv files in the directory structure and return their paths
    along with experiment info.
    """
    base_path = Path(base_dir)
    metrics_files = []
    
    # Walk through all experiment directories
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == 'mlruns':  # Skip mlruns directory
            continue
            
        experiment_name = exp_dir.name
        multiruns_dir = exp_dir / 'multiruns'
        
        if not multiruns_dir.exists():
            continue
            
        # Get the latest run directory (assuming format YYYY-MM-DD_HH-MM-SS)
        run_dates = [d for d in multiruns_dir.iterdir() if d.is_dir()]
        if not run_dates:
            continue
            
        latest_run = max(run_dates)
        
        # Find all run directories (0, 1, 2, etc.)
        for run_dir in latest_run.iterdir():
            if not run_dir.name.isdigit():
                continue
                
            run_number = int(run_dir.name)
            metrics_file = run_dir / 'csv' / 'version_0' / 'metrics.csv'
            
            if metrics_file.exists():
                metrics_files.append({
                    'path': metrics_file,
                    'experiment': experiment_name,
                    'run_number': run_number
                })
    
    return metrics_files


def create_plot(dfs_info, y_columns, title):
    """
    Create a plot combining metrics from multiple runs.
    dfs_info: List of tuples (DataFrame, experiment_name, run_number)
    """
    plt.figure(figsize=(12, 8))
    
    for df, exp_name, run_number in dfs_info:
        for y_column in y_columns:
            temp_df = df[['epoch', y_column]]
            temp_df = temp_df.dropna(subset=[y_column, 'epoch'])
            label = f"{exp_name} (Run {run_number}) - {y_column}"
            plt.plot(temp_df['epoch'], temp_df[y_column], label=label)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt


def main(logs_dir):
    # Find all metrics files
    metrics_files = find_metrics_files(logs_dir)
    
    if not metrics_files:
        print("No metrics files found!")
        return
    
    # Read all DataFrames and store with their metadata
    dfs_info = []
    for file_info in metrics_files:
        try:
            df = pd.read_csv(file_info['path'])
            dfs_info.append((
                df,
                file_info['experiment'],
                file_info['run_number']
            ))
        except Exception as e:
            print(f"Error reading {file_info['path']}: {e}")
    
    # Create plots
    plots = [
        (["val/acc"], "Training and Validation Accuracy"),
        (["val/loss"], "Training and Validation Loss"),
    ]
    
    # Create output directory
    output_dir = Path("training_plots")
    output_dir.mkdir(exist_ok=True)
    
    for y_columns, title in plots:
        chart = create_plot(dfs_info, y_columns, title)
        output_file = output_dir / f"{title.replace(' ', '_').lower()}_plot.png"
        chart.savefig(output_file, bbox_inches='tight')
        print(f"Generated plot: {output_file}")
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_plots.py <path_to_logs_folder>")
        sys.exit(1)

    logs_dir = sys.argv[1]
    if not os.path.exists(logs_dir):
        print(f"Error: Directory {logs_dir} does not exist.")
        sys.exit(1)

    main(logs_dir)