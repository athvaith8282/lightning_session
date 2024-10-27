import pandas as pd
import sys
import os
import matplotlib.pyplot as plt  # Importing matplotlib


def create_plot(df, y_columns, title):
    plt.figure()  # Create a new figure

    for y_column in y_columns:
        temp_df = df[['epoch', y_column]]
        temp_df = temp_df.dropna(subset=[y_column, 'epoch'])
        plt.plot(temp_df['epoch'], temp_df[y_column], label=y_column)  # Plot each y_column against 'epoch'
    
    plt.title(title)  # Set the title of the plot
    plt.xlabel('Epoch')  # Label for x-axis
    plt.ylabel(title)  # Label for y-axis
    plt.legend()  # Show legend
    return plt  # Return the plot object


def main(metrics_file):
    # Read the CSV file
    df = pd.read_csv(metrics_file)

    plots = [
        (["train/acc", "val/acc"], "Training and Validation Accuracy"),
        (["train/loss", "val/loss"], "Training and Validation Loss"),
    ]

    for y_columns, title in plots:
        chart = create_plot(df, y_columns, title)
        output_file = f"{title.replace(' ', '_').lower()}_plot.png"
        chart.savefig(output_file)  # Save the plot to a file
        print(f"Generated plot: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_plots.py <path_to_metrics_csv>")
        sys.exit(1)

    metrics_file = sys.argv[1]
    if not os.path.exists(metrics_file):
        print(f"Error: File {metrics_file} does not exist.")
        sys.exit(1)

    main(metrics_file)
