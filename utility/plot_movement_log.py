import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re


def extract_timestamp(filename):
    match = re.search(r'_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return None


def plot_movement_log(csv_path):

    # Load CSV
    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV is empty!")
        return

    # Extract time from filename
    df['timestamp'] = df['filename'].apply(extract_timestamp)

    # Normalize time so that first timestamp is 0, and convert to seconds
    first_time = df['timestamp'].min()
    df['time_sec'] = (df['timestamp'] - first_time) / 1000.0  # assuming timestamps are in milliseconds

    track_ids = df['track_id'].unique()
    print(f"Found {len(track_ids)} unique track_ids: {track_ids}")

    for track_id in track_ids:
        df_track = df[df['track_id'] == track_id]

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Movement analysis - Track ID {track_id}", fontsize=16)

        # Subplot 1: delta_pos
        plt.subplot(3, 1, 1)
        plt.plot(df_track['time_sec'], df_track['delta_pos'], label='delta_pos', color='blue')
        plt.ylabel('Delta Position (px)')
        plt.grid(True)
        plt.legend()

        # Subplot 2: delta_area
        plt.subplot(3, 1, 2)
        plt.plot(df_track['time_sec'], df_track['delta_area'], label='delta_area', color='orange')
        plt.ylabel('Delta Area (relative)')
        plt.grid(True)
        plt.legend()

        # Subplot 3: moving binary flag
        plt.subplot(3, 1, 3)
        plt.step(df_track['time_sec'], df_track['moving'], where='post', label='Moving', color='green')
        plt.ylabel('Moving (0/1)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()

        # Show plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot movement_log.csv with time axis")
    parser.add_argument('--csv', required=True, help='Path to movement_log.csv')

    args = parser.parse_args()
    plot_movement_log(args.csv)
