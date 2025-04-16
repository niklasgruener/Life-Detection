from ultralytics import YOLO

# Custom run names and their corresponding batch sizes
run_configs = [
    #("train_yolov8_1", 64),
    #("train_yolov8_2", 32),
    ("train_yolov8_back", 16)
]

for run_name, batch_size in run_configs:
    model = YOLO('yolov8n.pt')  # Load YOLOv11n base model

    for name, param in model.model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    print("Backbone layers frozen.")
    
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=batch_size,
        device=0,
        name=run_name
    )


#from ultralytics import YOLO


#model = YOLO('yolov8n.pt') # load pretrained YOLO model

#model.train(
#    data = 'data.yaml',
#    epochs = 50,
#    imgsz=640,
#    batch=16,
#    device=0
#)