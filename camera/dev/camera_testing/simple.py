from camera_core import Camera, Image
import cv2
import time

# Create a camera object
camera = Camera(bus_addr=[1,7])
camera2 = Camera(bus_addr=[1,8])


camera.warmup()
camera2.warmup()

camera.start()
camera2.start()

cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 640, 480)

cv2.namedWindow("Yolo2", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo2", 640, 480)

size = (1919,972)

#video1 = cv2.VideoWriter('left.mp4',  
#                         cv2.VideoWriter.fourcc(*'MP4V'), 
#                         20, size)

#video2 = cv2.VideoWriter('right.mp4',  
#                         cv2.VideoWriter.fourcc(*'MP4V'), 
#                         20, size)

while camera.stream:
    frame : Image = camera.get_latest_frame(undistort=True, with_cuda=True)
    frame2 : Image = camera2.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None or frame2 is None:
        continue
    frame = frame.frame
    frame2 = frame2.frame
#    video1.write(frame)
#    video2.write(frame2)

    cv2.imshow("Yolo", frame)
    cv2.imshow("Yolo2", frame2)
    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

camera.stop()
camera2.stop()
#video1.release()
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
