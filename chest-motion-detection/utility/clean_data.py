import os
import re
import shutil

# CONFIGURATION
top_source_root = 'chest-floor'    # contains jacket1/, polo1/, sweather1/
top_target_root = 'chest-floor-cleaned' # where to write the cleaned folders

timestamp_cutoff_start_ms = 43 * 1000  # 43 seconds in milliseconds
max_duration_ms = 82 * 1000            # maximal duration after first frame = 82 seconds

modalities = ['rgb', 'thermal', 'depth']
pattern = re.compile(r'_(\d{6})_(\d{10})\.png$')

# Find all clothing folders in top_source_root
clothing_folders = [d for d in os.listdir(top_source_root) if os.path.isdir(os.path.join(top_source_root, d))]

print(f"Found clothing folders: {clothing_folders}")

# Process each clothing folder
for clothing_folder in clothing_folders:
    print(f"\nProcessing clothing type: {clothing_folder}")

    source_clothing_path = os.path.join(top_source_root, clothing_folder)
    target_clothing_path = os.path.join(top_target_root, clothing_folder)
    os.makedirs(target_clothing_path, exist_ok=True)

    # Process each modality in this clothing folder
    for modality in modalities:
        print(f"\n  Processing modality: {modality}")

        source_folder = os.path.join(source_clothing_path, modality)
        target_folder = os.path.join(target_clothing_path, modality)
        os.makedirs(target_folder, exist_ok=True)

        # Collect matching files ≥ 43s
        files_with_timestamps = []
        for filename in os.listdir(source_folder):
            match = pattern.search(filename)
            if match:
                frame_id = int(match.group(1))
                timestamp_ms = int(match.group(2))
                if timestamp_ms >= timestamp_cutoff_start_ms:
                    files_with_timestamps.append((filename, timestamp_ms))

        if not files_with_timestamps:
            print(f"    No files found with timestamp ≥ 43 seconds in {modality}!")
            continue

        # Sort by timestamp
        files_with_timestamps.sort(key=lambda x: x[1])

        first_timestamp_ms = files_with_timestamps[0][1]

        print(f"    First timestamp to copy: {first_timestamp_ms} ms")

        # Keep only files within 82 seconds after first kept frame
        selected_files = []
        for filename, timestamp_ms in files_with_timestamps:
            relative_timestamp_ms = timestamp_ms - first_timestamp_ms
            if relative_timestamp_ms <= max_duration_ms:
                selected_files.append((filename, timestamp_ms))
            else:
                break  # stop when exceeding max duration

        if not selected_files:
            print(f"    No files remaining within max duration in {modality}!")
            continue

        print(f"    Keeping frames up to relative timestamp {max_duration_ms} ms")

        # Copy and rename files
        for new_frame_id, (filename, timestamp_ms) in enumerate(selected_files):
            relative_timestamp_ms = timestamp_ms - first_timestamp_ms
            new_filename = f"{modality}_{new_frame_id:06d}_{relative_timestamp_ms:010d}.png"
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(target_folder, new_filename)
            shutil.copyfile(src_path, dst_path)
            print(f"    Copied: {filename} → {new_filename}")

        # Count skipped frames at start
        skipped_start = len([
            f for f in os.listdir(source_folder)
            if pattern.search(f) and int(pattern.search(f).group(2)) < timestamp_cutoff_start_ms
        ])

        skipped_end = len(files_with_timestamps) - len(selected_files)
        total_copied = len(selected_files)

        print(f"    Summary for {modality}: copied {total_copied} frames, skipped {skipped_start} at start, skipped {skipped_end} at end.")

print("\nAll done.")

