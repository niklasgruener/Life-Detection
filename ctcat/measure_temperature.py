import cv2
import numpy as np


from utils import apply_custom_colormap, celsius_from_raw


from sensor import sensor



window_name = "Thermal"
cv2.namedWindow(window_name)

# state to remember mouse position
mouse_pos = {'x': 0, 'y': 0}

# callback to update mouse_pos on movement
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos['x'], mouse_pos['y'] = x, y


cv2.setMouseCallback(window_name, on_mouse)


while True:
    _, _, thermal_frame = next(sensor)

    temp_img = celsius_from_raw(thermal_frame)
    disp = apply_custom_colormap(thermal_frame, 'inferno')


    mx, my = mouse_pos['x'], mouse_pos['y']
    if 0 <= my < temp_img.shape[0] and 0 <= mx < temp_img.shape[1]:
        hover_t = temp_img[my, mx]
        hover_label = f"{hover_t:.1f} C"
        # draw a small background box for readability
        cv2.rectangle(disp,
                      (mx+10,   my+5),
                      (mx+10 +  (len(hover_label)*8), my+5+20),
                      (0,0,0), -1)
        cv2.putText(disp, hover_label,
                    (mx+5, my+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)

    # display
    cv2.imshow(window_name, disp)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        # ESC or 'q' to quit
        break