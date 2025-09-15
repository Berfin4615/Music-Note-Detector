import cv2
import numpy as np

# Image is uploaded from local path:
image_path = "/home/berfin/Desktop/Projects/Music-Note-Detector/türküler/ordunundereleri.png"
img = cv2.imread(image_path)
if img is None:
    print("Image not found.")
    exit()

# Grayscale conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Noise reduction
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding (inverse)
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
)

# Contour detection
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Filter and draw circular shapes resembling notes
note_count = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    # Filter based on aspect ratio and size
    if 0.7 < aspect_ratio < 1.3 and 10 < w < 40 and 10 < h < 40:
        note_count += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Note {note_count}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

print(f"Total note count: {note_count}")

# Show the result image
cv2.imshow("Detected Notes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
