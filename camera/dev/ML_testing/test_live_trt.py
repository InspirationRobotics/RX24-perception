import os

# Set YOLOv8 to quiet mode
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List
import cv2
import numpy as np
import time
from temp_undistort import Undistort

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine", task="detect", verbose=False)

# Load the undistortion
undistort = Undistort("/home/inspiration/RX24-perception/camera/dev/calibration/calib_img/camera_intrinsic_matrix.txt", "/home/inspiration/RX24-perception/camera/dev/calibration/calib_img/camera_distortion_matrix.txt", 1920, 1080)

# Load the camera
cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")

# Set the window size
cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 1280, 720)

prev_frame_time = 0
new_frame_time = 0

# Run inference
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break
    
    # Undistort the frame
    frame = undistort.undistort(frame, with_cuda=True)

    # Run inference
    results: list[Results] = trt_model(frame)

    # Draw the bounding boxes
    for result in results:
        names = result.names
        for box in result.boxes:
            conf = box.conf.item()
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            cls_id = box.cls.item()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{names[cls_id]}: {conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write FPS on the top left corner
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Yolo", frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()

# Note: Run this: export LD_PRELOAD=/lib/aarch64-linux-gnu/libstdc++.so.6:$LD_PRELOAD