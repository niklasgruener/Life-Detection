import os
import cv2
import numpy as np
from ultralytics import YOLO

def process_folder(
    folder_path,
    output_folder,
    model_path='yolo11.pt',
    results_file='results.txt',
    min_consecutive=6,
    pos_threshold=5,
    shape_threshold=5,
    conf_threshold=0.25
):
    """
    Process a folder of images, track people using a YOLO model, and detect movement based on bounding box changes.

    Args:
        folder_path (str): Path to the folder containing input images.
        output_folder (str): Path to save annotated output images.
        model_path (str): Path to the YOLO weights file.
        results_file (str): Filename for the output results text file.
        min_consecutive (int): Minimum number of consecutive frames with movement to tag the sequence.
        pos_threshold (float): Minimum pixel shift in bounding box center to count as movement.
        shape_threshold (float): Minimum pixel change in width or height to count as movement.
        conf_threshold (float): Confidence threshold for YOLO detections (0-1).
    """
    # Load model
    model = YOLO(model_path)

    # Gather and sort image paths
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    file_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(img_exts)
    ])
    if not file_paths:
        raise ValueError(f"No images found in folder: {folder_path}")

    # Read images
    images = []
    for p in file_paths:
        img = cv2.imread(p)
        if img is None:
            raise IOError(f"Failed to read image: {p}")
        images.append(img)

    # Run tracking on loaded images with confidence threshold
    results = model.track(
        source=images,
        conf=conf_threshold,
        show=False,
        save=False,
        persist=True
    )

    os.makedirs(output_folder, exist_ok=True)
    prev_boxes = {}
    movement_flags = []

    for idx, res in enumerate(results):
        img = images[idx].copy()
        h_img, w_img = img.shape[:2]
        max_area = (w_img * h_img) / 3

        # filter out overly large detections
        valid_boxes = []
        curr_boxes = {}
        for box in res.boxes:
            raw_id = box.id
            if raw_id is None:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().reshape(-1)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                # skip tracking big boxes
                continue
            tid = int(raw_id)
            curr_boxes[tid] = (x1, y1, x2, y2)
            valid_boxes.append(box)

        # detect movement based on filtered boxes
        movement = False
        for tid, coords in curr_boxes.items():
            if tid in prev_boxes:
                x1, y1, x2, y2 = coords
                px1, py1, px2, py2 = prev_boxes[tid]
                # center shift
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                dist = np.hypot(cx - pcx, cy - pcy)
                # size change
                w, h = x2 - x1, y2 - y1
                pw, ph = px2 - px1, py2 - py1
                dw = abs(w - pw)
                dh = abs(h - ph)
                if dist > pos_threshold or dw > shape_threshold or dh > shape_threshold:
                    movement = True
                    break

        movement_flags.append(movement)

        # draw filtered bounding boxes
        color = (0, 255, 0) if movement else (0, 0, 255)
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            id_text = f"ID:{int(box.id)}"
            cv2.putText(
                img,
                id_text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # save annotated frame
        out_path = os.path.join(output_folder, os.path.basename(file_paths[idx]))
        cv2.imwrite(out_path, img)

        prev_boxes = curr_boxes

    # determine overall movement tag
    consec = max_consec = 0
    for flag in movement_flags:
        if flag:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    folder_tag = 'movement' if max_consec >= min_consecutive else 'no movement'

    # write results
    with open(results_file, 'w') as f:
        f.write(f"{folder_tag}\n")
        for i, m in enumerate(movement_flags):
            f.write(f"frame {i}: {'movement' if m else 'no movement'}\n")

    print(f"Done: {folder_tag}. Results in {results_file}, annotated frames in {output_folder}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detect and record movement using YOLO tracking')
    parser.add_argument('--folder', required=True, help='Input image directory')
    parser.add_argument('--output', default='output', help='Annotated output directory')
    parser.add_argument('--model', default='yolo11.pt', help='YOLO weights file')
    parser.add_argument('--results', default='results.txt', help='Results text file')
    parser.add_argument('--min_consecutive', type=int, default=6, help='Min consecutive frames for movement tag')
    parser.add_argument('--pos_thresh', type=float, default=5.0, help='Pixel threshold for bbox center movement')
    parser.add_argument('--shape_thresh', type=float, default=5.0, help='Pixel threshold for bbox size change')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold for YOLO detections (0-1)')
    args = parser.parse_args()

    process_folder(
        folder_path=args.folder,
        output_folder=args.output,
        model_path=args.model,
        results_file=args.results,
        min_consecutive=args.min_consecutive,
        pos_threshold=args.pos_thresh,
        shape_threshold=args.shape_thresh,
        conf_threshold=args.conf_thresh
    )
