from ultralytics import YOLO

# Custom run names and their corresponding batch sizes
run_configs = [
    ("train_yolov9_1", 64),
    ("train_yolov9_2", 32),
    ("train_yolov9_3", 16)
]

for run_name, batch_size in run_configs:
    model = YOLO('yolov9t.pt')  # Load YOLOv11n base model
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=batch_size,
        device=0,
        name=run_name
    )


############YOLOV8############
#from ultralytics import YOLO


#model = YOLO('yolov8n.pt') # load pretrained YOLO model

#model.train(
#    data = 'data.yaml',
#    epochs = 50,
#    imgsz=640,
#    batch=16,
#    device=0
#)