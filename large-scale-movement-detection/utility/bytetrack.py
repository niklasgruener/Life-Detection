from ultralytics import YOLO

model = YOLO("/home/ngruener/thesis/life_detection/NEW/yolo/thermal/runs/detect/thermal_yolo11n/weights/best.pt")


results = model.track(
    task = "track",
    tracker = "bytetrack.yaml",
    source = "outdoor-night-thermal",
    save = True,
    save_txt = True,
    classes = [0],
    conf = 0.5,
    iou = 0.5,
    show = False
)


