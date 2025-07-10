import argparse
import cv2
import numpy as np
import os

def colorize_image(image_path, modality, output_path=None):
    # Load the image as 16-bit grayscale
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    if img.dtype != np.uint16:
        raise ValueError(f"Expected a 16-bit image. Got dtype: {img.dtype}")

    # Normalize to 8-bit for colormap application
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Select colormap
    if modality == 'thermal':
        colormap = cv2.COLORMAP_INFERNO
    elif modality == 'depth':
        colormap = cv2.COLORMAP_WINTER
    else:
        raise ValueError(f"Unknown modality: {modality}. Use 'thermal' or 'depth'.")

    # Apply colormap
    color_img = cv2.applyColorMap(img_8bit, colormap)

    # Define output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_{modality}_colorized.png"

    # Save the colorized image
    cv2.imwrite(output_path, color_img)
    print(f"Saved colorized image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize 16-bit thermal or depth image.")
    parser.add_argument("image", help="Path to the input 16-bit image")
    parser.add_argument("modality", choices=["thermal", "depth"], help="Modality of the image")
    parser.add_argument("--output", help="Optional output file path")

    args = parser.parse_args()

    colorize_image(args.image, args.modality, args.output)
