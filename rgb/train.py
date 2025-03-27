from ultralytics import YOLO


model = YOLO('yolov8n.pt') # load pretrained YOLO model

model.train(
    data = 'data.yaml',
    epochs = 50,
    imgsz=640,
    batch=32,
    device=0
)