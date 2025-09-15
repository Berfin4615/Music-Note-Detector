from ultralytics import YOLO

model = YOLO("/home/berfin/Desktop/Projects/Music-Note-Detector/runs/detect/train5/weights/best.pt")
results = model("/home/berfin/Desktop/Projects/Music-Note-Detector/türküler/ordunundereleri.png")  
results[0].save()  
