# 🇳🇵 Nepal Embossed License Plate Recognition System (YOLOv8 + OCR)
## 🟢 Project is live on https://nepal-embossed-license-plate-recognition.streamlit.app/

![Demo](assets/demo_output.png)

An **end-to-end license plate recognition system** designed specifically for **Nepal vehicle plates**, integrating:
- **YOLOv8** for license plate detection
- **OCR (EasyOCR)** for text extraction
- **Rule-based post-processing** to correct common OCR errors

This project demonstrates a **real-world AI pipeline** from detection → extraction → validation → usable output.

---

## 🚗 Pipeline Overview

Input Image
↓
YOLOv8 License Plate Detection
↓
Plate Cropping & Preprocessing
↓
OCR (EasyOCR)
↓
Post-processing & Plate Correction
↓
Final License Plate Text


---

## 📊 Detection Model Performance (YOLOv8n)

| Metric | Score |
|------|------|
| Precision | **0.9861** |
| Recall | **0.9494** |
| mAP@0.50 | **0.9772** |
| mAP@0.50:0.95 | **0.7196** |

---

## 📦 Dataset
Dataset is hosted on **Roboflow** and automatically downloaded during training/evaluation.

🔗 https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

> Dataset is **not included** in this repository.

---

## 🧠 Features
- Nepal-specific license plate correction logic
- Handles common OCR confusions (O/0, B/8, S/5, etc.)
- Modular and reusable pipeline
- Streamlit-ready for live demos

---

## 🗂 Project Structure

nepal-license-plate-recognition/
│
├── weights/best.pt
├── src/
│   ├── train.py
│   ├── evaluate_detection.py
│   ├── detect_and_crop.py
│   ├── ocr_plate.py
│   └── pipeline.py
│
├── streamlit/
│   └── app.py
├── data/sample_images/
├── assets/
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/GaneshNeupane01/nepal-embossed-license-plate-recognition.git
cd nepal-license-plate-recognition
```

---

The default scripts use example images from `data/sample_images/`.

---

# How to Run

## Option 1: Docker → Streamlit UI (recommended if you have Docker)

Build the image:

```bash
docker build -t platedetector .
```

Run the container and expose the Streamlit port:

```bash
docker run -p 8501:8501 platedetector
```

Then open in your browser:

```text
http://localhost:8501
```

The UI allows you to:

- Upload an image (JPG/PNG)
- Run YOLOv8 to detect the plate region
- Draw a bounding box on the detected plate
- Run OCR + correction and display the final plate number

---

## Option 2: Local Environment (without Docker)

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate  # Windows
```

Install dependencies (NumPy is already pinned in `requirements.txt` for PyTorch compatibility):

```bash
pip install -r requirements.txt
```

If you do not already have a compatible CPU build of PyTorch and torchvision, you can install them explicitly (CPU-only example):

```bash
pip install --no-cache-dir \
	torch==2.1.2+cpu torchvision==0.16.2+cpu \
	-f https://download.pytorch.org/whl/torch_stable.html
```

If you see a "Numpy is not available" error from PyTorch, make sure you are using a NumPy version `< 2.0` (already specified in `requirements.txt`).

---

### Option 2.1: Direct Streamlit UI

A minimal Streamlit app is provided in `streamlit/app.py`.

Run from the project root:

```bash
streamlit run streamlit/app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

**Note:** The app expects the model weights at `weights/best.pt`.

---

### Option 2.2: CLI (Detection + OCR)

There is a simple script in `src/pipeline.py` that will:

- Detect the plate with YOLOv8
- Crop it and apply preprocessing
- Run EasyOCR
- Post-process the text using Nepal-specific rules

Example (uses a default sample image if no argument is given):

```bash
python3 src/pipeline.py
```

To run on your own image:

```bash
python3 src/pipeline.py /path/to/your/image.jpg
```

## ☁️ Deploying on Hugging Face Spaces (Docker)

This repo includes a `Dockerfile` configured for Hugging Face Spaces.

Key points:

- Uses `python:3.10-slim`
- Installs all dependencies from `requirements.txt`
- Exposes port `8501`
- Starts Streamlit with:

	```bash
	streamlit run streamlit/app.py --server.port 8501 --server.address 0.0.0.0
	```

To deploy:

1. Create a new Space on Hugging Face
	 - **SDK**: Docker
	 - **Hardware**: any CPU space is enough for basic demo (GPU optional)
2. Push this project (including `weights/best.pt` and `Dockerfile`) to the Space repo.
3. The Space will build the Docker image and start the Streamlit app automatically.

Once the build is complete, open the Space URL and you will see the upload interface and predicted plate number with bounding box.

---

## 🧪 Sample Input / Output

- Example detected plate: `BAA4777`
- Screenshots are available in the `assets/` folder.

---

## 🚀 Future Work

- Character segmentation + character recognition (CNN)
- OCR fine-tuning for Nepali fonts
- Multi-plate detection support
- Real-time video inference

---

## 🧩 Tech Stack

- Python
- YOLOv8 (Ultralytics)
- EasyOCR
- OpenCV
- Roboflow
- Streamlit

---

## 👤 Author

Ganesh Neupane  
Computer Engineering | AI / ML



