import cv2
import os
from pathlib import Path


cap = cv2.VideoCapture("v4l2src device=/dev/video4 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")

CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / "star"

if not IMG_FILE_PATH.exists():
    os.mkdir(IMG_FILE_PATH)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Camera", 640, 480)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        continue

    cv2.imshow("Camera", frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break

    elif k == ord('s'):
        i += 1
        file_name = f"image_{i}.jpg"
        print(f'saving image: {file_name}')
        cv2.imwrite(str(IMG_FILE_PATH / file_name), frame)

    if i == 10:
        print("Max number of images saved")
        break

cap.release()
cv2.destroyAllWindows()