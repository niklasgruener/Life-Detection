import os
import argparse
import csv
import pandas as pd

import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import cm


def normalize(frame):
    """Normalize by static number - convert from float16 to uint8"""
    if (frame.ndim == 3):
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
    pos_threshold=5,
    size_threshold=5,
    colormap = None
):

    # load model and filenames
    model = YOLO(model_path)
    file_paths = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    )
    if not file_paths:
        raise ValueError(f'No images found in folder: {folder_path}')

    # create an output diretory, if not exists already
    os.makedirs(output_folder, exist_ok=True)


    prev_boxes = {}          # last-frame boxes {track_id: (cx,cy,w,h)}
    movement_log = []

    # load frame one by one (to avoid memory overflow)
    for idx, img_path in enumerate(file_paths):

        # read image
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f'Failed to read image: {img_path}')
        

        result = model.track(
            source=normalize(raw),
            persist=True,   # keep tracker state across successive calls
            tracker='bytetrack.yaml',
            classes=0,
            device=0
        )[0]


        colored = apply_custom_colormap(raw, colormap) if colormap else raw.copy()

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

            # Motion detection
            moving = False
            if track_id in prev_boxes:
                pcx, pcy, pw, ph = prev_boxes[track_id]
                delta_pos = np.hypot(cx - pcx, cy - pcy)
                delta_size = abs(w - pw) + abs(h - ph)

                # Check threshold
                if delta_pos >= pos_threshold or delta_size >= size_threshold:
                        moving = True

            # Update memory
            prev_boxes[track_id] = (cx, cy, w, h)

            # log
            movement_log.append({
                'frame': idx,
                'track_id': track_id,
                'moving': int(moving)
            })

            # Draw results 
            label = f"Victim {track_id} {'(Moving)' if moving else '(Still)'}"
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
        
        out_path = os.path.join(output_folder, f"{idx}.png")
        cv2.imwrite(out_path, colored)


    csv_path = os.path.join(output_folder, 'movement_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','track_id','moving'])
        writer.writeheader()
        writer.writerows(movement_log)
    print(f"Movement log saved to {csv_path}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and record movement using YOLO tracking')
    parser.add_argument('--folder', required=True, help='Input image directory')
    parser.add_argument('--output', default='output', help='Annotated output directory')
    parser.add_argument('--model', default='yolo11n.pt', help='YOLO weights file')
    parser.add_argument('--pos_thresh', type=float, default=10.0, help='Pixel threshold for bbox center movement')
    parser.add_argument('--size_thresh', type=float, default=10.0, help='Pixel threshold for bbox size change')
    parser.add_argument('--colormap', default=None, help='Colormap for colorizing thermal and depth images')

    args = parser.parse_args()

    eval(
        folder_path=args.folder,
        output_folder=args.output,
        model_path=args.model,
        pos_threshold=args.pos_thresh,
        size_threshold=args.size_thresh,
        colormap=args.colormap
    )
