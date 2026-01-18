import os
import sys

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Make src/ importable so we can reuse the existing pipeline logic
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from pipeline import _run_pipeline_on_plate_image  # type: ignore


@st.cache_resource(show_spinner=False)
def load_model():
    weights_path = os.path.join(PROJECT_ROOT, "weights", "best.pt")
    return YOLO(weights_path)


def detect_plate_and_ocr(image_np: np.ndarray):
    """Run detection + OCR, returning (plate_text, bbox, annotated_image).

    bbox is (x1, y1, x2, y2) in the original image coordinates.
    """
    model = load_model()

    results = model(image_np)[0]
    if len(results.boxes) == 0:
        return None, None, image_np

    # Take the first detected plate
    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])

    # Crop plate region
    plate_img = image_np[y1:y2, x1:x2]

    # Run OCR pipeline on cropped plate
    plate_text = _run_pipeline_on_plate_image(plate_img)

    # Draw bounding box on a copy for visualization
    annotated = image_np.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return plate_text, (x1, y1, x2, y2), annotated


def main():
    st.title("Embossed License Plate Detector")
    st.write("Upload an image and the app will detect the plate region, draw a bounding box, and run OCR to read the plate number.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Detect Plate"):
            with st.spinner("Running detection and OCR..."):
                plate_text, bbox, annotated = detect_plate_and_ocr(image_np)

            if plate_text is None or bbox is None:
                st.error("No license plate could be detected.")
            else:
                st.success(f"Detected plate number: **{plate_text}**")

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if annotated.shape[2] == 3 else annotated
                st.image(annotated_rgb, caption="Detected plate region", use_column_width=True)


if __name__ == "__main__":
    main()
