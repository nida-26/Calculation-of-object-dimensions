import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/runs/detect/train3/weights/best.pt")
# Set the dataset location
dataset_location = "C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/datasets"

# Set the confidence threshold
conf = 0.25

# Predict on the test images
results = model.predict(
    source=f"{dataset_location}/train/images",
    conf=conf,
    save=True
)