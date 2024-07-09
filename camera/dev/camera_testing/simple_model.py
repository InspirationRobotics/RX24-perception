from camera_core import Camera, Image, ML_Model
import cv2
import time

# Create a camera object
# camera = Camera(video_path="/home/inspiration/RX24-perception/camera/dev/ML_testing/countdown.mp4")
camera = Camera()
camera1 = Camera(4)

model = ML_Model("/home/inspiration/RX24-perception/camera/camera_core/models/yolov8n.engine", "tensorrt")
model1 = ML_Model("/home/inspiration/RX24-perception/camera/camera_core/models/yolov8n.engine", "tensorrt")


camera.load_model_object(model)
camera1.load_model_object(model1)

camera.warmup()
camera1.warmup()

camera.start_stream()
camera.start_model()

camera1.start_stream()
camera1.start_model()

cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Yolo", 1280, 720)
pre_results = None

while camera.stream:

    pre_time = time.time()

    frame : Image = camera.get_latest_frame(undistort=True, with_cuda=True)
    frame1 : Image = camera1.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None:
        continue
    frame = frame.frame

    results = camera.get_latest_model_results()
    if results == pre_results:
        continue
    pre_results = results
    for result in results:
        names = result.names
        for box in result.boxes:
            conf = box.conf.item()
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            cls_id = box.cls.item()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{names[cls_id]}: {conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    post_time = time.time() - pre_time
    # print(f"Draw Time taken: {post_time :.4f}")

    # Write FPS on the top left corner
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - pre_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Yolo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop_model()
camera1.stop_model()

camera.stop_stream()
camera1.stop_stream()

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
