import argparse
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import csv

def normalize(frame: np.ndarray) -> np.ndarray:
    """Per-frame min–max normalization to 8-bit BGR"""
    f32 = frame.astype(np.float32)
    mn, mx = f32.min(), f32.max()
    scaled = (f32 - mn) / (mx - mn + 1e-10)
    gray_scaled = (scaled * 255).astype(np.uint8)
    return cv2.cvtColor(gray_scaled, cv2.COLOR_GRAY2BGR)

def apply_custom_colormap(frame: np.ndarray, cmap_name: str) -> np.ndarray:
    """Normalize and apply a matplotlib colormap, return BGR image."""
    f16 = frame.astype(np.float16)
    mn, mx = f16.min(), f16.max()
    norm = (f16 - mn) / (mx - mn + 1e-10)
    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

def get_default_roi(frame_shape: tuple) -> tuple:
    """Calculate and return a default central ROI based on frame dimensions."""
    h, w = frame_shape[:2]
    x = int(w * 0.25)
    y = int(h * 0.40)
    rw = int(w * 0.50)
    rh = int(h * 0.30)
    return (x, y, rw, rh)

def process_frame(frame: np.ndarray, modality: str) -> tuple:
    """
    Process a single frame based on modality and return:
    (raw_bgr, colored_bgr, gray)
    """
    if modality == 'depth':
        raw = normalize(frame)
        colored = apply_custom_colormap(frame, 'winter')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    elif modality == 'rgb':
        raw = frame.copy()
        colored = frame.copy()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    elif modality == 'thermal':
        raw = normalize(frame)
        colored = apply_custom_colormap(frame, 'inferno')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid modality: {modality}")
    return raw, colored, gray

def compute_optical_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two consecutive grayscale frames."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, roi: tuple) -> np.ndarray:
    """Draw semi-transparent flow arrows on the ROI and return the blended frame."""
    x, y, rw, rh = roi
    overlay = frame.copy()
    step = 16
    arrow_scale = 10.0
    arrow_color = (0, 0, 0)
    thickness = 1
    tip_length = 0.3

    for j in range(y, y + rh, step):
        for i in range(x, x + rw, step):
            dx, dy = flow[j, i]
            start_pt = (i, j)
            end_pt = (int(i + arrow_scale * dx), int(j + arrow_scale * dy))
            cv2.arrowedLine(
                overlay,
                start_pt,
                end_pt,
                arrow_color,
                thickness=thickness,
                tipLength=tip_length,
                line_type=cv2.LINE_AA
            )

    alpha = 0.4
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

def main():
    parser = argparse.ArgumentParser(description="Process frames from a folder for respiratory detection")
    parser.add_argument(
        "modality",
        choices=["rgb", "depth", "thermal"],
        help="Modality of the frames in the input folder"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the folder containing input frames"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the folder where processed frames and plot will be saved"
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Region of interest (x y width height). If omitted, a default central ROI is used"
    )
    parser.add_argument(
        "--arrows",
        action="store_true",
        help="Enable drawing semi-transparent flow arrows (disabled by default)"
    )
    args = parser.parse_args()

    modality = args.modality
    input_dir = args.input_dir
    output_dir = args.output_dir
    draw_arrows = args.arrows
    roi = tuple(args.roi) if args.roi is not None else None

    os.makedirs(output_dir, exist_ok=True)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    file_paths = []
    for ext in extensions:
        file_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    file_paths.sort()  # Ensure consistent ordering

    if not file_paths:
        print(f"No images found in {input_dir}")
        return

    prev_gray = None
    signal = []
    timestamps = []
    frame_index = 0

    for file_path in file_paths:
        # === Extract timestamp ===
        base_name = os.path.basename(file_path)
        parts = base_name.split('_')
        if len(parts) < 3:
            print(f"Warning: unexpected filename format: {base_name}")
            continue

        timestamp_str = parts[2].split('.')[0]
        timestamp = int(timestamp_str)
        timestamp_seconds = timestamp / 1e3  # Use milliseconds → seconds
        timestamps.append(timestamp_seconds)

        # === Read frame ===
        if modality == "rgb":
            frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                tmp = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                frame = tmp if tmp is not None else None

        if frame is None:
            print(f"Warning: could not read {file_path}, skipping.")
            continue

        raw, colored, gray = process_frame(frame, modality)

        # Determine ROI if not specified
        if roi is None:
            roi = get_default_roi(gray.shape)
        x, y, rw, rh = roi

        # Draw bounding box
        cv2.rectangle(colored, (x, y), (x + rw, y + rh), (0, 255, 0), 2)

        if prev_gray is not None:
            flow = compute_optical_flow(prev_gray, gray)
            roi_flow = flow[y : y + rh, x : x + rw]
            avg_y = float(np.mean(roi_flow[..., 1]))
            signal.append(avg_y)

            if draw_arrows:
                display_frame = draw_flow_arrows(colored, flow, roi)
            else:
                display_frame = colored.copy()
        else:
            display_frame = colored.copy()

        # Save processed frame
        save_path = os.path.join(output_dir, base_name)
        cv2.imwrite(save_path, display_frame)

        prev_gray = gray.copy()
        frame_index += 1

    # === Plot ===
    if signal and len(timestamps) > 1:
        timestamps_plot = timestamps[1:]  # Skip first timestamp

        plt.figure(figsize=(8, 4))
        plt.plot(timestamps_plot, signal, linestyle='-', linewidth=0.6, color='tab:blue')

        plt.title(f"Respiratory Signal ({modality})", fontsize=12)
        plt.xlabel("Time (seconds)", fontsize=11)
        plt.ylabel("Average Vertical Flow (pixels/frame)", fontsize=11)

        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=10))
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())

        plt.tight_layout(pad=2)

        plot_path = os.path.join(output_dir, "signal_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        
        # Save vertical flow data to CSV
        csv_path = os.path.join(output_dir, "signal_data.csv")
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["time_seconds", "average_vertical_flow"])
            for t, s in zip(timestamps_plot, signal):
                writer.writerow([t, s])
        
        print(f"Processed {frame_index} frames. Signal plot saved to {plot_path}")
    else:
        print("No optical-flow signal was generated (perhaps only one frame).")

if __name__ == "__main__":
    main()

