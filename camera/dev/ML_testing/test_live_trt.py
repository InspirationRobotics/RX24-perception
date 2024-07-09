import os

# Set YOLOv8 to quiet mode
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List
import cv2
import numpy as np
import time
from camera_core import Camera, Image

# Create a camera object
camera = Camera()
camera1 = Camera(4)
camera2 = Camera(8)

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine", task="detect", verbose=False)

# Run inference on a single sample frame to warm up the model
warmup_frame = cv2.imread("sample.jpg")
warmup_frame = camera.warmup_undistort(warmup_frame)
trt_model(warmup_frame)

# Load the camera
camera.start_stream()
camera1.start_stream()
camera2.start_stream()

# Set the window size
cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 1280, 720)

prev_frame_time = 0
new_frame_time = 0
true_start = time.time()
# Run inference
while camera.stream and camera1.stream and camera2.stream:

    # Calculate FPS
    prev_frame_time = time.time()
    
    # Get the latest frame
    frame : Image = camera.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None:
        continue
    frame = frame.frame
    # Get the latest frame
    frame1 : Image = camera1.get_latest_frame(undistort=True, with_cuda=True)
    if frame1 is None:
        continue
    frame1 = frame1.frame
    # Get the latest frame
    frame2 : Image = camera2.get_latest_frame(undistort=True, with_cuda=True)
    if frame2 is None:
        continue
    frame2 = frame2.frame
    # Run inference
    pre_time = time.time()
    results: list[Results] = trt_model(frame)
    results1: list[Results] = trt_model(frame1)
    results2: list[Results] = trt_model(frame2)
    post_time = time.time() - pre_time
    print(f"Inference Time taken: {post_time:.4f}")

    pre_time = time.time()
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
    post_time = time.time() - pre_time
    # print(f"Draw Time taken: {post_time :.4f}")

    # Write FPS on the top left corner
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Yolo", frame)
    
    # Exit if 'q' is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        print("Exiting...")
        break

print(f"Total time taken: {time.time() - true_start:.4f}")
camera.stop_stream()
camera1.stop_stream()
camera2.stop_stream()
cv2.destroyAllWindows()

# Note: Run this: export LD_PRELOAD=/lib/aarch64-linux-gnu/libstdc++.so.6:$LD_PRELOAD
# 2.5 second delay with or without undistort (TODO retest this with new suite)
# Not doing undistort boosts FPS by about 10, from 23fps to 33fps