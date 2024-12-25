from ultralytics import YOLO

# Load the YOLOv8s model
model = YOLO("yolov8s.pt")

# Set the dataset location
dataset_location = "C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/datasets"

# Train the model
results = model.train(
    data=f"{dataset_location}/data.yaml",
    epochs=25,
    imgsz=800,
    plots=True
)