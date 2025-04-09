from flask import Flask, Response, render_template
from markupsafe import escape
import cv2
import time

#from ultralytics import YOLO

# Web server
app = Flask(__name__)


# YOLO
#model_rgb = YOLO('rgb.pt')
#model_depth = YOLO('depth.pt')
#model_thermal = YOLO('thermal.pt')



cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = 1 / fps if fps > 0 else 1 / 30


#def run_yolo(frame, model):
#    return model.predict(source=frame, save=False)



def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(delay)


#def run_motion_detection():
#    frame = None
#    model = None
#    results = run_yolo(frame, model)


#def run_thermal_detection():
#   frame = None
#   results = run_yolo(frame, model_thermal)


@app.route('/detect-motion')
def detect_motion():
    return render_template('detect_motion.html')

@app.route('/detect-thermal')
def detect_thermal():
    return render_template('detect_thermal.html')


@app.route("/detect-motion-stream")
def detect_motion_stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/detect-thermal-stream")
def detect_thermal_stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
