import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# Initialize YOLOv5
model = YOLO("depth/runs/detect/train3/weights/best.pt")

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Store previous centers for each track
track_history = defaultdict(list)

# Movement threshold (pixels)
MOVEMENT_THRESHOLD = 5

# Load input video
input_path = "input_video_depth.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for .avi
out = cv2.VideoWriter("output_video_depth.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]
    detections = []

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        cx, cy = (l + w) / 2, (t + h) / 2

        # Append current center to history
        track_history[track_id].append((cx, cy))

        # Determine movement
        color = (0, 0, 255)  # Red (no motion)
        label = "Still"
        if len(track_history[track_id]) >= 2:
            x_prev, y_prev = track_history[track_id][-2]
            dist = np.sqrt((cx - x_prev) ** 2 + (cy - y_prev) ** 2)
            if dist > MOVEMENT_THRESHOLD:
                color = (0, 255, 0)  # Green
                label = "Moving"

        # Draw on frame
        cv2.rectangle(frame, (int(l), int(t)), (int(w), int(h)), color, 2)
        cv2.putText(frame, f"ID {track_id} - {label}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print("Video processing complete. Saved as 'output_video.mp4'")
