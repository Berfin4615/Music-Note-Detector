import cv2
import numpy as np

image_path = "/home/berfin/Desktop/Projects/Music-Note-Detector/türküler/ordunundereleri.png"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive threshold to highlight staff lines
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
)

# Morphological open to isolate horizontal lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find Y positions of detected lines
line_y_positions = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h <= 4 and w > 100:
        line_y_positions.append(y)

# Cluster lines that are close in Y (e.g. ±4px)
line_y_positions = sorted(line_y_positions)
grouped_lines = []
threshold = 4

for y in line_y_positions:
    if not grouped_lines:
        grouped_lines.append([y])
    elif abs(y - grouped_lines[-1][-1]) <= threshold:
        grouped_lines[-1].append(y)
    else:
        grouped_lines.append([y])

# Average Y position for each line
average_lines = [int(np.mean(group)) for group in grouped_lines]

# Draw full horizontal lines
for y in average_lines:
    cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)

cv2.imwrite("full_staff_lines.jpg", img)
print("🎼 Tüm portre çizgileri baştan sona çizildi ve 'full_staff_lines.jpg' olarak kaydedildi.")
