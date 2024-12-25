import os
import glob
from IPython.display import Image, display
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/runs/detect/train3/weights/best.pt")

# Set the confidence threshold
conf = 0.25

for image_path in glob.glob('C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/runs/detect/predict/*.jpg')[:3]:
    results = model.predict(source=image_path, conf=conf, save=True)
    display(Image(filename=image_path, width=600))
    print("\n")

    # Iterate over the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x, y, width, height = box.xywh  # Get the box coordinates in [x, y, width, height] format
            print(f"Object at ({x}, {y}) with width {width} and height {height}")
        boxes = result.boxes
        for box in boxes:
            x, y, width, height = box.xywh  # Get the box coordinates in [x, y, width, height] format
            print(f"Object at ({x}, {y}) with width {width} and height {height}")