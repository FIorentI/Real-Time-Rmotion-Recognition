import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import PIL
import io

# Load the YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('Yolo_Weights\\yolov8m_200e.pt')  #  YOLO('yolov8n_100e.pt') for nano model
model.to(device)

threshold = 0.5  # confidence threshold for detection


# Function to detect objects in the image
def detect_image(img):
    results = model.predict(img, verbose=False)
    images, boxes, conf, clss = [], [], [], []

    for i, result in enumerate(results):
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()

        for j in range(len(bboxes)):
            (x1, y1, x2, y2), score, c = bboxes[j], scores[j], cls[j]
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if score >= threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (int(c == 0) * 255, int(c == 1) * 255, int(c == 2) * 255), 2)
                boxes.append(bbox)
                clss.append(c)
                conf.append(score)
        images.append(img)

    return images, boxes, conf, clss


# Function to display FPS and overlay bounding boxes
def process_frame(frame, total_fps):
    start = time.time()

    # Detect objects in the frame
    _, boxes, confs, clss = detect_image(frame)

    # FPS Calculation
    fps = 1 / (time.time() - start)
    total_fps.append(fps)

    # Overlay bounding boxes and FPS on the original frame
    for box, cls, score in zip(boxes, clss, confs):
        left, top, right, bottom = box
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        # Add confidence score above the bounding box
        cv2.putText(frame, f"Score: {score:.2f}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add FPS information on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Avg. FPS: {np.mean(total_fps):.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Max. FPS: {max(total_fps):.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Min. FPS: {min(total_fps):.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return frame


# Start webcam capture
cap = cv2.VideoCapture(0)  # Open the webcam (0 is the default camera)
total_fps = []

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Process frame and overlay bounding boxes and FPS directly on it
    processed_frame = process_frame(frame.copy(), total_fps)

    # Display the resulting frame with bounding boxes and FPS overlay
    cv2.imshow('Object Detection - YOLO', processed_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
