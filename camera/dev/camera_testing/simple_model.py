import os
os.environ['YOLO_VERBOSE'] = 'False'

from camera_core import Camera, Image, ML_Model
import cv2
import time

# Create a camera object
# camera = Camera(video_path="/home/inspiration/RX24-perception/camera/dev/ML_testing/countdown.mp4")
camera = Camera(bus_addr=(1,5))
camera1 = Camera(bus_addr=(1,6))

full_engine = '/home/inspiration/RX24-perception/camera/dev/ML_testing/yolov8n.engine'
half_engine = '/home/inspiration/RX24-perception/camera/camera_core/models/yolov8n.engine'
model = ML_Model(full_engine, "tensorrt")
model1 = ML_Model(full_engine, "tensorrt")


camera.load_model_object(model)
camera1.load_model_object(model1)

camera.warmup()
camera1.warmup()

camera.start()
camera1.start()

cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 1280, 720)
cv2.namedWindow("Yolo2", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo2", 1280, 720)
pre_results = None

while camera.stream:

    pre_time = time.time()

    frame : Image = camera.get_latest_frame(undistort=True, with_cuda=True)
    frame1 : Image = camera1.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None or frame1 is None:
        continue
 
    frame = frame.frame
    frame1 = frame1.frame

    frame = camera.draw_model_results(frame)
    frame1 = camera1.draw_model_results(frame1)
    post_time = time.time() - pre_time
    # print(f"Draw Time taken: {post_time :.4f}")

    # Write FPS on the top left corner
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - pre_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if frame is not None:
        cv2.imshow("Yolo", frame)
    if frame1 is not None:
        cv2.imshow("Yolo2", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
camera1.stop()

# video_path="/home/inspiration/RX24-perception/camera/dev/ML_testing/countdown.mp4"
# cap = cv2.VideoCapture(video_path)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: failed to capture image")
#         break
#     frame = frame
#     print("got frame")
#     #cv2.imshow("Yolo", frame)
#     #k = cv2.waitKey(1) & 0xFF
#     # if k == ord('q'):
#     #     print("Exiting...")
#     #     break
#     time.sleep(1/30)

# cap.release()
