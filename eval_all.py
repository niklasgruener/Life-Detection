import os
import subprocess

BASE_DIR = "scenarios/indoor-day-near-solo/activities"
MODEL_PATH = "models/yolo11n_depth.pt"
COLORMAP = "winter"

# Loop through all subdirectories in activities/
for activity in os.listdir(BASE_DIR):
    activity_path = os.path.join(BASE_DIR, activity)
    modality_path = os.path.join(activity_path, "depth")

    if os.path.isdir(modality_path):
        output_dir = os.path.join(activity_path, "output", "depth")
        os.makedirs(output_dir, exist_ok=True)

        command = [
            "python",
            "eval_motion_detection.py",
            "--output", output_dir,
            "--folder", modality_path,
            "--model", MODEL_PATH,
            "--colormap", COLORMAP
        ]

        print("Running:", " ".join(command))
        subprocess.run(command)
