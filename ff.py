import cv2
import numpy as np
from tkinter import filedialog, Tk

# Hide Tkinter root window
Tk().withdraw()

# File selection dialog
file_path = filedialog.askopenfilename(
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")],
    title="Select an Image for Fire Detection"
)

if not file_path:
    print("User canceled the image selection.")
    exit()

# Read the image
img = cv2.imread(file_path)
cv2.imshow("Original Image", img)

# Convert BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

# Normalize to [0,1] for comparison
h = h / 179.0   # OpenCV hue range: 0–179
s = s / 255.0
v = v / 255.0

# Fire mask based on HSV thresholds
fire_mask = ((h >= 0) & (h <= 0.1)) & (s >= 0.4) & (v >= 0.5)
fire_mask = fire_mask.astype(np.uint8) * 255

# Morphological opening to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

# Count fire pixels
fire_pixels = np.sum(fire_mask > 0)
fire_threshold = 500

if fire_pixels > fire_threshold:
    print(f"🔥 Fire detected! Number of fire pixels: {fire_pixels}")
else:
    print(f"No significant fire detected. Number of fire pixels: {fire_pixels}")

cv2.imshow("Fire Mask", fire_mask)

# Create overlay
overlay = img.copy()
overlay[fire_mask > 0] = [0, 0, 255]   # Highlight fire regions in red

cv2.imshow("Fire Detection Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
