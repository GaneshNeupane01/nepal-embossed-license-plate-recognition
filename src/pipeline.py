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
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)

    ocr_results = reader.readtext(eroded, paragraph=False)
    if not ocr_results:
        print("❌ OCR failed to find text")
        return None

    box_data = []
    full_text = ''
    for (bbox, text, conf) in ocr_results:
        if conf < 0.2: continue # Ignore very low confidence
        full_text += text + " "

    full_text = full_text.strip().upper()
    print(f"OCR detected text: {full_text}")
    cleaned = clean_input(full_text)
    print(f"Cleaned OCR text: {cleaned}")
    final_plate = correct_plate(cleaned)      


    return final_plate


def run_pipeline_on_image(image):
    """Run the full pipeline on an in-memory image (numpy array).

    Returns the detected plate string or None.
    """
    plate_img = detect_plate(image)
    if plate_img is None:
        return None

    return _run_pipeline_on_plate_image(plate_img)


def run_pipeline(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    return run_pipeline_on_image(image)


if __name__ == "__main__":
    plate = run_pipeline("data/sample_images/sample1.png")
    print("Detected Plate:", plate)
