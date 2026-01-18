from ultralytics import YOLO
from roboflow import Roboflow
import os

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

WORKSPACE = "roboflow-universe-projects"
PROJECT = "license-plate-recognition-rxg4e"
VERSION = 11

MODEL_PATH = "weights/best.pt"

# Load Dataset
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
dataset = rf.workspace(WORKSPACE) \
             .project(PROJECT) \
             .version(VERSION) \
             .download("yolov8")


# Load Model
model = YOLO(MODEL_PATH)

metrics = model.val(
    data=f"{dataset.location}/data.yaml",
    split="val"
)

print("\nValidation Metrics")
print("-" * 30)
print(f"Precision     : {metrics.box.mp:.4f}")
print(f"Recall        : {metrics.box.mr:.4f}")
print(f"mAP@0.50      : {metrics.box.map50:.4f}")
print(f"mAP@0.50:0.95 : {metrics.box.map:.4f}")
