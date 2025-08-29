# 🔥 Fire Detection Using Python (OpenCV)

This project detects **fire regions** in an image using **color-based HSV filtering** and highlights them with a red overlay.  
It is a Python implementation of the original MATLAB version.

---

## 📌 Features
- Select an image from your computer for analysis.
- Convert image from **RGB → HSV color space**.
- Apply thresholds on **Hue, Saturation, and Value** to detect fire-like regions.
- Morphological filtering to reduce noise.
- Overlay detected fire regions in **red** on the original image.
- Print number of detected fire pixels and decide whether **fire is present**.

---

## 🛠️ Technologies Used
- **Python 3.x**
- **OpenCV (cv2)**
- **NumPy**
- **Tkinter** (for file dialog)

---

## 📂 Project Structure
├── ff.py # Main Python script
├── fire.jpg # Example test image
└── README.md # Project documentation
