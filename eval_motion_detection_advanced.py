import os
import argparse
import csv
from collections import deque, defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from matplotlib import cm


def normalize(frame):
    """Normalize by static number - convert from float16 to uint8"""
    if frame.ndim == 3:
        return frame
    
    frame = np.asarray(frame, dtype=np.float16)
    scaled = (frame / 65535) * 255
    rgb = cv2.cvtColor(scaled.astype(np.uint8), cv2.COLOR_GRAY2BGR)
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
    pos_threshold=5,
    size_threshold=5,
    colormap=None,
    window_size=5,
    vote_thresh=3
):
    # Load model and gather image paths
    model = YOLO(model_path)
    file_paths = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.png')
    )
    if not file_paths:
        raise ValueError(f'No images found in folder: {folder_path}')

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Prepare CSV logging
    csv_path = os.path.join(output_folder, 'movement_log.csv')
    log_file = open(csv_path, 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow([
        'frame_idx', 'track_id', 'delta_pos', 'delta_size', 'moving_raw', 'moving_smooth'
    ])

    # History for smoothing (per-track)
    history = defaultdict(lambda: deque(maxlen=window_size))
    prev_boxes = {}

    # Process each frame
    for idx, img_path in enumerate(file_paths):
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f'Failed to read image: {img_path}')

        result = model.track(
            source=normalize(raw),
            persist=True,
            tracker='bytetrack.yaml',
            classes=0,
            device=0
        )[0]

        colored = apply_custom_colormap(raw, colormap) if colormap else raw.copy()

        for box in result.boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().ravel().astype(int)
            track_id = int(box.id.item())

            # Compute center and size
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Initialize deltas and raw movement flag
            delta_pos, delta_size = 0.0, 0.0
            moving_raw = False
            if track_id in prev_boxes:
                pcx, pcy, pw, ph = prev_boxes[track_id]
                delta_pos = np.hypot(cx - pcx, cy - pcy)
                delta_size = abs(w - pw) + abs(h - ph)
                moving_raw = (delta_pos >= pos_threshold or delta_size >= size_threshold)

            # Update previous box state
            prev_boxes[track_id] = (cx, cy, w, h)

            # Temporal smoothing: majority vote over last window_size frames
            history[track_id].append(int(moving_raw))
            moving_smooth = (sum(history[track_id]) >= vote_thresh)

            # Log to CSV
            writer.writerow([
                idx, track_id, f"{delta_pos:.2f}", f"{delta_size:.2f}",
                int(moving_raw), int(moving_smooth)
            ])

            # Draw box and label
            label = f"Victim {track_id} {'(Moving)' if moving_smooth else '(Still)'}"
            color = (0, 255, 0) if moving_smooth else (0, 0, 255)
            cv2.rectangle(colored, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                colored,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Save annotated frame
        out_path = os.path.join(output_folder, f"{idx}.png")
        cv2.imwrite(out_path, colored)

    # Close CSV log
    log_file.close()

    # Print summary of movement
    df = pd.read_csv(csv_path)
    frames_with_motion = sorted(df.loc[df.moving_smooth == 1, 'frame_idx'].unique())
    print('Frames with ANY movement (smoothed):', frames_with_motion)
    for vid, group in df.groupby('track_id'):
        mv = sorted(group.loc[group.moving_smooth == 1, 'frame_idx'].unique())
        print(f"Victim {vid} moved in frames: {mv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect and record movement using YOLO tracking with temporal smoothing'
    )
    parser.add_argument('--folder', required=True, help='Input image directory')
    parser.add_argument('--output', default='output', help='Annotated output directory')
    parser.add_argument('--model', default='yolo11.pt', help='YOLO weights file')
    parser.add_argument(
        '--pos_thresh', type=float, default=10.0,
        help='Pixel threshold for bbox center movement'
    )
    parser.add_argument(
        '--size_thresh', type=float, default=10.0,
        help='Pixel threshold for bbox size change'
    )
    parser.add_argument(
        '--colormap', default=None,
        help='Colormap for colorizing thermal and depth images'
    )
    parser.add_argument(
        '--window', type=int, default=5,
        help='Number of frames for smoothing window'
    )
    parser.add_argument(
        '--votes', type=int, default=3,
        help='Minimum "moving" votes in window to declare motion'
    )

    args = parser.parse_args()
    eval(
        folder_path=args.folder,
        output_folder=args.output,
        model_path=args.model,
        pos_threshold=args.pos_thresh,
        size_threshold=args.size_thresh,
        colormap=args.colormap,
        window_size=args.window,
        vote_thresh=args.votes
    )
