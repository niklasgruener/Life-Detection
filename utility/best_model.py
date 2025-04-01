import os
import pandas as pd

# List of training run directories
run_dirs = ['../rgb/runs/detect/train', '../rgb/runs/detect/train2', '../rgb/runs/detect/train3']

best_run = None
best_overall_value = float('-inf')
best_metrics = None

# Loop over each run directory
for run in run_dirs:
    csv_path = os.path.join(run, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"results.csv not found in {run}")
        continue

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Check if the expected metric column exists
    if 'metrics/mAP50-95(B)' not in df.columns:
        print(f"'metrics/mAP50-95(B)' column not found in {csv_path}")
        continue

    # Find the row with the highest metrics/mAP50-95(B) for this run
    idx_best = df['metrics/mAP50-95(B)'].idxmax()
    best_row = df.loc[idx_best]
    current_value = best_row['metrics/mAP50-95(B)']
    print(f"{run} best mAP50-95(B): {current_value}")

    # Update the overall best if this run's best is higher than previous ones
    if current_value > best_overall_value:
        best_overall_value = current_value
        best_run = run
        best_metrics = best_row

# Display results for the best overall model
if best_metrics is not None:
    print(f"\nOverall best model found in {best_run} with metrics/mAP50-95(B): {best_overall_value}\n")
    # Convert the best row into a DataFrame for a table view
    best_model_df = best_metrics.to_frame().transpose()
    print("Metrics for the best model:")
    print(best_model_df)
    
    # Save the best model metrics to a CSV file
    output_csv = "best_model_results.csv"
    best_model_df.to_csv(output_csv, index=False)
    print(f"\nBest model metrics saved to {output_csv}")
else:
    print("No valid results found across the run directories.")
