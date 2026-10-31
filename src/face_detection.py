
"""OpenCV face detection utilities."""
from pathlib import Path
import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def detect_faces(image):
    """Return (x, y, w, h) boxes for all detected faces in a BGR/RGB image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image supplied for face detection.")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {CASCADE_PATH}")
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

def extract_face(image, box):
    x, y, w, h = [int(v) for v in box]
    h_img, w_img = image.shape[:2]
    x, y = max(0, x), max(0, y)
    crop = image[y:min(y+h, h_img), x:min(x+w, w_img)]
    if crop.size == 0:
        raise ValueError("Detected face region is empty.")
    return crop
