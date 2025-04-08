import cv2
import numpy as np

# Calibration parameters 
# Here, raw values are in hundredths of Kelvin:
#   0°C = 273.15 K  => raw offset 27315
# and every 100 raw units corresponds to 1°C.
THERMAL_RAW_OFFSET = 27315  # raw value for 0°C
THERMAL_RAW_SCALE = 100     # raw units per °C

# File name of the thermal image (assumed to be in 16-bit format).
image_filename = 'assets/thermal.png'

# Load the thermal image as a 16-bit image.
raw_image = cv2.imread(image_filename, cv2.IMREAD_ANYDEPTH)
if raw_image is None:
    raise FileNotFoundError("Thermal image file not found.")

celsius_image = (raw_image.astype(np.float32) - THERMAL_RAW_OFFSET) / THERMAL_RAW_SCALE

# For display: normalize the temperature range.
# Define the expected temperature range (in °C) for visualization.
temp_min = 20.0   # adjust this as needed
temp_max = 40.0   # adjust this as needed

# Clip temperatures to the expected range and normalize to 0-1.
normalized_image = np.clip((celsius_image - temp_min) / (temp_max - temp_min), 0, 1)
# Convert normalized values to 8-bit for color mapping.
normalized_uint8 = (normalized_image * 255).astype(np.uint8)

# Apply a color map for visualization. (Using COLORMAP_JET here, can try COLORMAP_HOT, etc.)
color_image = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)

# Callback function to update displayed temperature when moving the mouse.
def mouse_callback(event, x, y, flags, param):
    # 'param' is a dictionary containing the Celsius image and the base colorized image.
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the temperature at the current mouse location.
        temp = param["celsius_image"][y, x]
        # Create a copy of the base colorized image.
        img_copy = param["color_image"].copy()
        # Prepare the text string.
        text = f"{temp:.1f} °C"
        # Choose a location near the mouse pointer to overlay the text.
        # You can adjust the font scale and thickness as needed.
        cv2.putText(img_copy, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # Update the window with the temporary image.
        cv2.imshow("Thermal Image", img_copy)

# Package the necessary images into a dict for the callback.
callback_params = {"celsius_image": celsius_image, "color_image": color_image}

# Create the window and assign the mouse callback.
cv2.namedWindow("Thermal Image")
cv2.setMouseCallback("Thermal Image", mouse_callback, callback_params)

# Display the color-mapped thermal image.
cv2.imshow("Thermal Image", color_image)
print("Hover your mouse over the thermal image window to inspect temperatures (in °C).")

# Wait until a key is pressed, then exit.
cv2.waitKey(0)
cv2.destroyAllWindows()
