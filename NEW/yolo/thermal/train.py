from ultralytics import YOLO


model = YOLO('yolo11n.pt') # load pretrained YOLO model

model.train(
    data = 'data.yaml',
    epochs = 50,
    imgsz=640,
    batch=64,
    device=0,
    name="thermal_yolo11n"
)
