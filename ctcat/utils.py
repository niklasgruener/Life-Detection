import cv2
import numpy as np
import matplotlib.cm as mpl_cm  # for colormap functions

THERMAL_RAW_OFFSET = 27315  # raw value corresponding to 0°C
THERMAL_RAW_SCALE  = 100     # raw units per °C


def normalize(frame: np.ndarray) -> np.ndarray:
    """Convert a raw sensor frame to 8-bit BGR."""
    frame_f = np.asarray(frame, dtype=np.float16)
    scaled = ((frame_f / 65535) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    return rgb


def apply_custom_colormap(frame: np.ndarray, cmap_name: str) -> np.ndarray:
    """Normalize and apply a matplotlib colormap, return BGR image."""
    f32 = frame.astype(np.float32)
    mn, mx = f32.min(), f32.max()
    norm = (f32 - mn) / (mx - mn + 1e-10)
    cmap = mpl_cm.get_cmap(cmap_name)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def celsius_from_raw(raw: np.ndarray) -> np.ndarray:
    """Convert raw thermal units to Celsius."""
    return (raw.astype(np.float32) - THERMAL_RAW_OFFSET) / THERMAL_RAW_SCALE

    