import sys
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import cm


from ctcat import CTCAT_Sensor, CTCAT_DataFormat
from status import SaveStatusIndication, StatusColor, StatusMode


def normalize(frame):
    """Normalize by static number - convert from float16 to uint8"""
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


def load_model(modality):
    return YOLO(f"models/{modality}.pt")


def run_track(modality='rgb', pos_threshold=0.03, area_threshold=0.05, debounce_window=5, debounce_required=3):
    """Detect and Track victims using yolo model in the given modality."""
    
    # Load YOLO model
    model = load_model(modality)

    # Configure sensor
    cfg = {
        'device': 'linux-arm64',
        'sensor_id': 3
    }

    sensor = CTCAT_Sensor(
            sensor_id=cfg['sensor_id'], 
            data_format=CTCAT_DataFormat.Resized,
            colorize=False, 
            fps=None, 
            device=cfg['device']
            )


    # Tracking memory
    prev_boxes = {}
    prev_movement_history = {}
 

    while True:
        # Get the next frame from the sensor
        depth_frame, color_frame, thermal_frame = next(sensor)

        # Select the frame based on the chosen modality
        if modality == 'depth':
            raw = normalize(depth_frame) 
            colored = apply_custom_colormap(depth_frame, 'winter')
        elif modality == 'rgb':
            raw = color_frame
            colored = color_frame
        elif modality == 'thermal':
            raw = normalize(thermal_frame)
            colored = apply_custom_colormap(thermal_frame, 'inferno')
        else:
            print(f"Invalid modality selected: {sensor_modality}")
            break


        # Run inference on the selected frame
        result = model.track(source=raw, persist=True, tracker="bytetrack.yaml",classes=0, device=0)[0]  

        # Display results
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

            

            # Motion detection
            moving = False
            raw_moving = False
            if track_id in prev_boxes:
                pcx, pcy, pw, ph = prev_boxes[track_id]
                prev_area = pw * ph

                # Relative position change
                delta_pos_raw = np.hypot(cx - pcx, cy - pcy)
                box_diag = np.hypot(w, h)
                delta_pos = delta_pos_raw / (box_diag + 1e-5)

                # Area change 
                delta_area = abs(curr_area - prev_area) / (prev_area + 1e-5)

                # Check threshold
                if delta_pos >= pos_threshold or delta_area >= area_threshold:
                    raw_moving = True

            print(raw_moving)
            # Update memory
            prev_boxes[track_id] = (cx, cy, w, h)

            # Update history for debouncing
            history = prev_movement_history.get(track_id, [])
            history.append(int(raw_moving))
            if len(history) > debounce_window:
                history.pop(0)
            prev_movement_history[track_id] = history

            # Apply debouncing decision
            num_moving = sum(history)
            moving = num_moving >= debounce_required
            
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
        
        cv2.imshow("Tracking Victims", colored)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    print("\nFinished.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run victim detection and motion tracking.")

    parser.add_argument("modality", choices=["rgb", "depth", "thermal"], help="Sensor modality to use")
    parser.add_argument("--pos-threshold", type=float, default=0.03, help="Position change threshold (default: 10.0)")
    parser.add_argument("--area-threshold", type=float, default=0.05, help="Size change threshold (default: 10.0)")

    args = parser.parse_args()

    run_track(args.modality, args.pos_threshold, args.area_threshold)

