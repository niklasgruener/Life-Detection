import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Ensure you have a SORT implementation installed
from collections import defaultdict

# Initialize YOLO
model = YOLO("thermal/runs/detect/train/weights/best.pt")

# Initialize SORT tracker
tracker = Sort(max_age=30)

# Store previous centers for each track
track_history = defaultdict(list)
# Store bounding box shape history
bbox_history = defaultdict(list)

# Movement and shape change thresholds (pixels)
MOVEMENT_THRESHOLD = 5
SHAPE_THRESHOLD = 5

# Load input video
input_path = "input_video2_thermal.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("track_video3_thermal.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]
    detections = []

    for box in results.boxes.data:
        # YOLO returns [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        # Create detection list for SORT: [x1, y1, x2, y2, conf]
        detections.append([x1, y1, x2, y2, conf])

    # Convert detections to numpy array; handle case with no detections
    if detections:
        dets = np.array(detections)
    else:
        dets = np.empty((0, 5))

    # Update SORT tracker; returns an array of tracks: [x1, y1, x2, y2, track_id]
    tracks = tracker.update(dets)

    for track in tracks:
        l, t, r, b, track_id = track
        track_id = int(track_id)  # ensure track_id is an integer
        cx, cy = (l + r) / 2, (t + b) / 2
        current_width  = r - l
        current_height = b - t

        # Update histories
        track_history[track_id].append((cx, cy))
        bbox_history[track_id].append((current_width, current_height))

        # Determine movement based on center position and bounding box shape
        moving_pos = False
        moving_shape = False

        if len(track_history[track_id]) >= 2:
            x_prev, y_prev = track_history[track_id][-2]
            dist = np.sqrt((cx - x_prev) ** 2 + (cy - y_prev) ** 2)
            if dist > MOVEMENT_THRESHOLD:
                moving_pos = True

        if len(bbox_history[track_id]) >= 2:
            w_prev, h_prev = bbox_history[track_id][-2]
            if (abs(current_width - w_prev) > SHAPE_THRESHOLD or
                abs(current_height - h_prev) > SHAPE_THRESHOLD):
                moving_shape = True

        # Label based on movement
        if moving_pos or moving_shape:
            color = (0, 255, 0)
            label = "Moving"
        else:
            color = (0, 0, 255)
            label = "Still"

        # Draw bounding box and label
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
        cv2.putText(frame, f"ID {track_id} - {label}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Write processed frame to output video
    out.write(frame)

cap.release()
out.release()
print("Video processing complete. Saved as 'track_video_thermal.mp4'")
