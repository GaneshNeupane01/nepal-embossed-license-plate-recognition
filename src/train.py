from ultralytics import YOLO
from roboflow import Roboflow
import os

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
dataset = rf.workspace("roboflow-universe-projects") \
             .project("license-plate-recognition-rxg4e") \
             .version(11) \
             .download("yolov8")

model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
