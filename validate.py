from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/runs/detect/train3/weights/best.pt")

# Set the dataset location
dataset_location = "C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/datasets"

# Validate the model
results = model.val(data=f"{dataset_location}/data.yaml")