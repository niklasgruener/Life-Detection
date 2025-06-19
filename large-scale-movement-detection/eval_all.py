import os
import subprocess

BASE_DIR = "evaluation/activities"
MODEL_PATH = "models/yolo11n_thermal.pt"
COLORMAP = "inferno"

# Loop through all subdirectories in activities/
for activity in os.listdir(BASE_DIR):
    activity_path = os.path.join(BASE_DIR, activity)
    modality_path = os.path.join(activity_path, "thermal")

    if os.path.isdir(modality_path):
        output_dir = os.path.join("output", activity, "thermal")
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
