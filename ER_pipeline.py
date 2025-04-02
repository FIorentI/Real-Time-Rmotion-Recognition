import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image

# Load YOLO Object Detection Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('Yolo_Weights\\yolov8m_200e.pt')
yolo_model.to(device)

# Load Emotion Recognition Model
# Recreate the same model architecture
emotion_model = models.mobilenet_v2(pretrained=False)  # Don't load pretrained weights
emotion_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
emotion_model.classifier[1] = torch.nn.Linear(1280, 7)  # Adjust the classifier if needed

# Load the saved weights
emotion_model.load_state_dict(torch.load("Mobilenet\\FER_best_model.pth", map_location=device))

# Move model to the correct device and set to eval mode
emotion_model.to(device)
emotion_model.eval()

# Define emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define transformations
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees=90),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

threshold = 0.5  # Detection confidence threshold
total_fps = []


def detect_objects(img):
    results = yolo_model.predict(img, verbose=False)
    boxes, confs, clss = [], [], []

    for result in results:
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()

        for i in range(len(bboxes)):
            if scores[i] >= threshold:
                boxes.append(tuple(map(int, bboxes[i])))
                confs.append(scores[i])
                clss.append(cls[i])

    return boxes, confs, clss


def recognize_emotion(roi):
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = emotion_model(roi_tensor)
        predicted_emotion = EMOTION_LABELS[torch.argmax(output).item()]

    return predicted_emotion


def process_frame(frame):
    start = time.time()

    boxes, confs, clss = detect_objects(frame)

    for box in boxes:
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            emotion = recognize_emotion(roi)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    fps = 1 / (time.time() - start)
    total_fps.append(fps)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return frame


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame.copy())
    cv2.imshow('Object & Emotion Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
