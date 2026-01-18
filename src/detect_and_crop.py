import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("weights/best.pt")

def detect_plate(image_np):
    results = model(image_np)[0]

    if len(results.boxes) == 0:
        return None

    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
    return image_np[y1:y2, x1:x2]
