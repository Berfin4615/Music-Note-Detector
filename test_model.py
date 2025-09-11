from ultralytics import YOLO

model = YOLO("/home/berfin/Desktop/Projects/Music-Note-Detector/runs/detect/train5/weights/best.pt")
results = model("image.png")  
results[0].show()  
