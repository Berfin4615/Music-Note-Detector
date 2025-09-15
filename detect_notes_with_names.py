import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- CONFIG --------------------
IMAGE_PATH = '/home/berfin/Desktop/Projects/Music-Note-Detector/türküler/ordunundereleri.png'
MODEL_PATH = '/home/berfin/Desktop/Projects/Music-Note-Detector/runs/detect/train5/weights/best.pt'
OUTPUT_PATH = 'output_detected_notes.jpg'

# -------------------- LOAD IMAGE --------------------
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------------- DETECT STAFF LINES --------------------
def detect_staff_lines(gray_img):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    _, binary = cv2.threshold(detected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [cv2.boundingRect(c)[1] for c in contours]
    lines = sorted(lines)

    # Grupla ve 5 çizgiyi döndür
    grouped = []
    for y in lines:
        if not grouped or abs(grouped[-1] - y) > 4:
            grouped.append(y)
    return grouped[:5]

staff_lines = detect_staff_lines(gray)

# -------------------- NOTE MAPPING --------------------
def match_note_to_pitch(y, staff_lines):
    if len(staff_lines) != 5:
        return "?"
    notes_order = ['fa', 're', 'si', 'sol', 'mi', 'do', 'la', 'fa', 're', 'si']  # yukarıdan aşağı
    spacing = (staff_lines[4] - staff_lines[0]) / 4
    note_positions = [staff_lines[0] - i * (spacing / 2) for i in range(len(notes_order))]
    distances = [abs(y - pos) for pos in note_positions]
    idx = np.argmin(distances)
    return notes_order[idx].capitalize()

# -------------------- DETECT NOTES WITH YOLO --------------------
model = YOLO(MODEL_PATH)
results = model(image)[0]

for box in results.boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    if 'note' in class_name.lower():
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_y = int((y1 + y2) / 2)
        note_name = match_note_to_pitch(center_y, staff_lines)

        # Çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, note_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.circle(image, (int((x1 + x2)/2), center_y), 2, (255, 0, 0), -1)

# -------------------- SAVE --------------------
cv2.imwrite(OUTPUT_PATH, image)
print(f"✅ Saved to {OUTPUT_PATH}")