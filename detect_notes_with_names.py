from ultralytics import YOLO
import cv2
import os

model_path = "/home/berfin/Desktop/Projects/Music-Note-Detector/runs/detect/train5/weights/best.pt"  # kendi path'in buysa sorun yok
image_path = "image.png"  # test etmek istediğin görsel

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model bulunamadı: {model_path}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

model = YOLO(model_path)

img = cv2.imread(image_path)

results = model.predict(image_path)[0]  # ilk sonuç
print("✅ Tahmin tamamlandı.")

names = model.names  # {0: 'note', 1: 'clef'...}
if results.boxes is not None:
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf = box.conf[0]

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)

        # 🟩 KUTUYU ÇİZ
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 🔤 METNİ YAZ
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"{cls_name} ({conf:.2f}) - [{x1}, {y1}, {x2}, {y2}]")
else:
    print("⚠️ Hiç nesne tespit edilmedi.")

output_path = "output_detected_notes.jpg"
cv2.imwrite(output_path, img)
print(f"💾 Görsel kaydedildi: {output_path}")

try:
    cv2.imshow("Detected Notes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("⚠️ Görsel gösterilemedi (muhtemelen GUI desteği yok).")
