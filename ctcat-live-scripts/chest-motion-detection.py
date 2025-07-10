import argparse
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from ctcat import CTCAT_Sensor, CTCAT_DataFormat

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

def setup_sensor(sensor_id: int, device: str) -> CTCAT_Sensor:
    """Configure and return the CTCAT sensor."""
    return CTCAT_Sensor(
        sensor_id=sensor_id,
        data_format=CTCAT_DataFormat.Resized,
        colorize=False,
        fps=None,
        device=device
    )

def initialize_plot() -> tuple:
    """Set up Matplotlib for live plotting and return figure components."""
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-', lw=1)
    ax.set_title("Live Respiratory Signal (avg vertical flow)")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Average flow in ROI (y)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    return fig, ax, line

def get_default_roi(frame_shape: tuple) -> tuple:
    """Calculate and return a default central ROI based on frame dimensions."""
    h, w = frame_shape[:2]
    x = int(w * 0.25)
    y = int(h * 0.40)
    rw = int(w * 0.50)
    rh = int(h * 0.30)
    return (x, y, rw, rh)

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
    arrow_scale = 5.0
    arrow_color = (0, 0, 255)
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

def update_live_plot(ax, line, signal: list, frame_idx: int):
    """Update the Matplotlib line plot with new signal data."""
    line.set_data(np.arange(len(signal)), signal)
    if frame_idx > ax.get_xlim()[1]:
        ax.set_xlim(0, frame_idx + 20)
    cur_min, cur_max = np.min(signal), np.max(signal)
    buf = 0.1 * max(abs(cur_min), abs(cur_max), 1.0)
    ax.set_ylim(cur_min - buf, cur_max + buf)
    ax.relim()
    ax.autoscale_view(scaley=False)
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

def process_frame(modality: str, depth_frame: np.ndarray, color_frame: np.ndarray, thermal_frame: np.ndarray) -> tuple:
    """
    Process raw frames based on modality and return raw BGR, colored BGR,
    and grayscale images.
    """
    if modality == 'depth':
        raw = normalize(depth_frame)
        colored = apply_custom_colormap(depth_frame, 'winter')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    elif modality == 'rgb':
        raw = color_frame.copy()
        colored = color_frame.copy()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    elif modality == 'thermal':
        raw = normalize(thermal_frame)
        colored = apply_custom_colormap(thermal_frame, 'inferno')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid modality selected: {modality}")
    return raw, colored, gray

def run_detection(modality: str = 'rgb', roi: tuple = None, draw_arrows: bool = False):
    """Main loop for respiratory motion detection with live plotting and optional flow arrows."""
    sensor_config = {'device': 'linux-arm64', 'sensor_id': 3}
    sensor = setup_sensor(sensor_config['sensor_id'], sensor_config['device'])

    fig, ax, line = initialize_plot()
    frame_idx = 0
    prev_gray = None
    signal = []

    try:
        while True:
            depth_frame, color_frame, thermal_frame = next(sensor)
            raw, colored, gray = process_frame(modality, depth_frame, color_frame, thermal_frame)

            # Initialize ROI if not already specified
            if roi is None:
                roi = get_default_roi(gray.shape)
            x, y, rw, rh = roi
            cv2.rectangle(colored, (x, y), (x + rw, y + rh), (0, 255, 0), 2)

            if prev_gray is not None:
                flow = compute_optical_flow(prev_gray, gray)
                roi_flow = flow[y:y + rh, x:x + rw]
                avg_y = float(np.mean(roi_flow[..., 1]))
                signal.append(avg_y)

                if draw_arrows:
                    display_frame = draw_flow_arrows(colored, flow, roi)
                else:
                    display_frame = colored.copy()

                frame_idx += 1
                update_live_plot(ax, line, signal, frame_idx)
            else:
                display_frame = colored.copy()

            prev_gray = gray.copy()
            cv2.imshow("Respiratory Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show(block=True)
        print("\nFinished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run respiratory detection")
    parser.add_argument("modality", choices=["rgb", "depth", "thermal"], help="Sensor modality to use")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Region of interest for chest (x y width height in pixels). "
             "If omitted, a central default ROI is used."
    )
    parser.add_argument(
        "--arrows",
        dest="draw_arrows",
        action="store_true",
        help="Enable drawing semi-transparent flow arrows (disabled by default)."
    )

    args = parser.parse_args()
    roi = tuple(args.roi) if args.roi is not None else None
    run_detection(args.modality, roi, args.draw_arrows)

