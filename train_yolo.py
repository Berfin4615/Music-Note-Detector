from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(
    data="/home/berfin/Desktop/Projects/Music-Note-Detector/data/Music Notes detection.v1i.yolov8/data.yaml", 
    epochs=50,
    imgsz=640,
    batch=8,         
    patience=10,     
    device=0         
)