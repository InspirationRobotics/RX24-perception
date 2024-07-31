from camera_core import Camera, Image
import cv2
import time

# Create a camera object
camera = Camera(4)

camera.warmup()

camera.start()

cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 640, 480)

while camera.stream:
    frame : Image = camera.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None:
        continue
    frame = frame.frame
    cv2.imshow("Yolo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
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
