import os
import glob
from IPython.display import Image, display
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("C:/Users/nidaf/OneDrive/Documents/GitHub/Calculation-of-Object-Dimensions/runs/detect/train3/weights/best.pt")

# Set the confidence threshold
conf = 0.25

# Open the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Frame dimensions: {frame_width}x{frame_height}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame as a temporary image
    cv2.imwrite('temp.jpg', frame)

    # Predict on the temporary image
    results = model('temp.jpg')

    # Get the bounding box coordinates and class labels from the results
    for result in results:
        boxes = result.boxes.xyxy
        class_labels = result.names

        # Draw bounding boxes and display dimensions
        for box in boxes:
            x1, y1, x2, y2 = box
            conf = 0.5  # default confidence score
            cls = 0  # default class label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_labels[int(cls)]} ({conf:.2f})", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Width: {x2 - x1:.2f} pixels, Height: {y2 - y1:.2f} pixels", (int(x1), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Object Detection', frame)

    # Exit on key press
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()