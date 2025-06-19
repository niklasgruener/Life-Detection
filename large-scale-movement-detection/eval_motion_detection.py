import os
import argparse
import csv
import pandas as pd

import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import cm


##########################################################################################
# PLOTTING results
##########################################################################################

import matplotlib.pyplot as plt
import re

def extract_timestamp(filename):
    match = re.search(r'_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return None

def plot_movement_log(csv_path, output_folder):
    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV is empty!")
        return

    # Extract time from filename
    df['timestamp'] = df['filename'].apply(extract_timestamp)

    # Normalize time so first timestamp = 0
    first_time = df['timestamp'].min()
    df['time_sec'] = (df['timestamp'] - first_time) / 1000.0  # ms to sec

    track_ids = df['track_id'].unique()
    print(f"Plotting movement for {len(track_ids)} track_ids...")

    plots_folder = os.path.join(output_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    for track_id in track_ids:
        df_track = df[df['track_id'] == track_id]

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Movement analysis - Victim {track_id}", fontsize=16)

        # Subplot 1: delta_pos (relative now!)
        plt.subplot(3, 1, 1)
        plt.plot(df_track['time_sec'], df_track['delta_pos'], label='delta_pos', color='blue')
        plt.ylabel('Delta Position (relative)')
        plt.grid(True)

        # Subplot 2: delta_area
        plt.subplot(3, 1, 2)
        plt.plot(df_track['time_sec'], df_track['delta_area'], label='delta_area', color='orange')
        plt.ylabel('Delta Area (relative)')
        plt.grid(True)

        # Subplot 3: moving binary flag
        plt.subplot(3, 1, 3)
        plt.step(df_track['time_sec'], df_track['moving'], where='post', label='Moving', color='green')
        plt.ylabel('Moving (0/1)')
        plt.xlabel('Time (s)')
        plt.grid(True)

        # Save plot
        out_plot_path = os.path.join(plots_folder, f"movement_track_{track_id}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_plot_path)
        plt.close()

        print(f"Saved plot for track_id {track_id} to {out_plot_path}")


##########################################################################################
##########################################################################################
##########################################################################################

def normalize(frame):
    """Normalize by static number - convert from float16 to uint8"""
    if frame.ndim == 3:
        return frame
    
    frame = np.asarray(frame, dtype=np.float16)
    scaled = (frame/65535)*255 
    rgb = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR).astype(np.uint8)  
    return rgb


def apply_custom_colormap(frame: np.ndarray, cmap_name: str) -> np.ndarray:
    """Normalize and apply a matplotlib colormap, return BGR image."""
    f16 = frame.astype(np.float16)
    mn, mx = f16.min(), f16.max()
    norm = (f16 - mn) / (mx - mn + 1e-10)
    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def eval(
    folder_path,
    output_folder,
    model_path='yolo11.pt',
    pos_threshold=0.05,  # default adjusted for relative delta_pos
    area_threshold=0.05,
    colormap=None,
    debounce_window=3,
    debounce_required=2
):

    # Load model and filenames
    model = YOLO(model_path)
    file_paths = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    )
    if not file_paths:
        raise ValueError(f'No images found in folder: {folder_path}')

    # Create output directory if not exists
    os.makedirs(output_folder, exist_ok=True)

    prev_boxes = {}               # Last-frame boxes {track_id: (cx, cy, w, h)}
    prev_movement_history = {}    # Movement history per track_id
    movement_log = []

    # Process each frame one by one
    for idx, img_path in enumerate(file_paths):

        # Read image
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f'Failed to read image: {img_path}')
        
        # Run YOLO tracking
        result = model.track(
            source=normalize(raw),
            persist=True,   # Keep tracker state across successive calls
            tracker='bytetrack.yaml',
            classes=0,
            device=0
        )[0]

        # Apply colormap if specified
        colored = apply_custom_colormap(raw, colormap) if colormap else raw.copy()

        # Extract original filename
        original_name = os.path.basename(img_path)

        # Process each detected box
        for box in result.boxes:
            if box.id is None:
                continue

            coords = box.xyxy.cpu().numpy().ravel().astype(int)
            x1, y1, x2, y2 = coords
            track_id = int(box.id.item())

            # Compute center and size
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            curr_area = w * h

            # Motion detection using relative position + area change
            raw_moving = False
            delta_pos = 0.0
            delta_area = 0.0

            if track_id in prev_boxes:
                pcx, pcy, pw, ph = prev_boxes[track_id]
                prev_area = pw * ph

                # Relative position change
                delta_pos_raw = np.hypot(cx - pcx, cy - pcy)
                box_diag = np.hypot(w, h)
                delta_pos = delta_pos_raw / (box_diag + 1e-5)

                # Area change
                delta_area = abs(curr_area - prev_area) / (prev_area + 1e-5)

                # Check thresholds
                if delta_pos >= pos_threshold or delta_area >= area_threshold:
                    raw_moving = True

            # Update memory
            prev_boxes[track_id] = (cx, cy, w, h)

            # Update movement history for debouncing
            history = prev_movement_history.get(track_id, [])
            history.append(int(raw_moving))
            if len(history) > debounce_window:
                history.pop(0)
            prev_movement_history[track_id] = history

            # Apply debouncing decision
            num_moving = sum(history)
            moving = num_moving >= debounce_required

            # Log movement
            movement_log.append({
                'frame': idx,
                'filename': original_name,
                'track_id': track_id,
                'moving': int(moving),
                'delta_pos': delta_pos,  # now relative!
                'delta_area': delta_area,
                'debounce_window': history.copy(),
                'num_moving_in_window': num_moving
            })

            # Draw results
            label = f"Victim {track_id} {'(Moving)' if moving else ''}"
            color = (0, 255, 0) if moving else (0, 0, 255)

            cv2.rectangle(colored, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                colored,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Save annotated frame using original filename
        out_path = os.path.join(output_folder, original_name)
        cv2.imwrite(out_path, colored)

    # Save movement log to CSV
    csv_path = os.path.join(output_folder, 'movement_log.csv')
    fieldnames = [
        'frame',
        'filename',
        'track_id',
        'moving',
        'delta_pos',
        'delta_area',
        'debounce_window',
        'num_moving_in_window'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in movement_log:
            # Save debounce_window as string
            row['debounce_window'] = str(row['debounce_window'])
            writer.writerow(row)

    print(f"Movement log saved to {csv_path}")
    plot_movement_log(csv_path, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and record movement using YOLO tracking')
    parser.add_argument('--folder', required=True, help='Input image directory')
    parser.add_argument('--output', default='output', help='Annotated output directory')
    parser.add_argument('--model', default='yolo11n.pt', help='YOLO weights file')
    parser.add_argument('--pos_thresh', type=float, default=0.03, help='Relative threshold for bbox center movement (fraction of box diag)')
    parser.add_argument('--area_thresh', type=float, default=0.05, help='Relative threshold for bbox area change (e.g., 0.05 = 5%)')
    parser.add_argument('--colormap', default=None, help='Colormap for colorizing thermal and depth images')
    parser.add_argument('--debounce_window', type=int, default=5, help='Debounce window size (number of frames)')
    parser.add_argument('--debounce_required', type=int, default=3, help='Number of frames in window required to classify as moving')

    args = parser.parse_args()

    eval(
        folder_path=args.folder,
        output_folder=args.output,
        model_path=args.model,
        pos_threshold=args.pos_thresh,
        area_threshold=args.area_thresh,
        colormap=args.colormap,
        debounce_window=args.debounce_window,
        debounce_required=args.debounce_required
    )
