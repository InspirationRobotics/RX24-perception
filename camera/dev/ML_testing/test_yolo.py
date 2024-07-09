import os
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO
import numpy as np
import cv2
import time
from camera_core import Camera, Image

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Create a camera object
camera = Camera()

warmup_frame = cv2.imread("sample.jpg")
warmup_frame = camera.warmup_undistort(warmup_frame)
model(warmup_frame)

camera.start_stream()

#cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")
# cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 1280, 720)

pre_frame = 0
post_frame = 0

while camera.stream:
    pre_frame = time.time()
    frame : Image = camera.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None:
        continue
    frame = frame.frame
    results = model.predict(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, result.names[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    post_frame = time.time()
    fps = 1 / (post_frame - pre_frame)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop_stream()
cv2.destroyAllWindows()