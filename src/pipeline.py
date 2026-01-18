import cv2
import easyocr
import numpy as np
from PIL import Image
from detect_and_crop import detect_plate
from ocr_plate import clean_text, correct_plate

reader = easyocr.Reader(['en'], gpu=False)


def clean_input(text: str) -> str:
    """Wrapper around clean_text for backward compatibility."""
    return clean_text(text)


def _run_pipeline_on_plate_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)

    # detail=1 -> includes confidence; paragraph must be bool, not string
    ocr_results = reader.readtext(eroded, detail=1, paragraph=False)
    if not ocr_results:
        print("❌ OCR failed to find text")
        return None, None

    full_text = ""
    confidences = []

    for item in ocr_results:
        # Handle different shapes from EasyOCR
        if len(item) == 3:
            bbox, text, conf = item
        elif len(item) == 2:
            bbox, rest = item
            if isinstance(rest, (list, tuple)) and len(rest) == 2:
                text, conf = rest
            else:
                text, conf = rest, None
        else:
            continue

        # Filter by confidence if we have it
        if conf is not None and conf < 0.2:
            continue

        full_text += text + " "
        if conf is not None:
            confidences.append(conf)

    full_text = full_text.strip().upper()
    if not full_text:
        return None, None

    cleaned = clean_input(full_text)
    final_plate = correct_plate(cleaned)


    return final_plate


def run_pipeline_on_image(image):
    """Run the full pipeline on an in-memory image (numpy array).

    Returns (plate_string, confidence or None).
    """
    plate_img = detect_plate(image)
    if plate_img is None:
        return None, None

    return _run_pipeline_on_plate_image(plate_img)


def run_pipeline(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    return run_pipeline_on_image(image_path)


if __name__ == "__main__":
    plate, conf = run_pipeline("data/sample_images/test.jpg")
    print("Detected Plate:", plate, "Conf:", conf)