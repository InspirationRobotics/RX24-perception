import cv2
import numpy as np
from camera_core import Undistort

cap = cv2.VideoCapture("v4l2src device=/dev/video4 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL) 

cv2.resizeWindow("Undistorted", 640, 480)
cv2.resizeWindow("Original", 640, 480)

# load the camera intrinsic matrix and distortion coefficients
intrinsics = np.loadtxt('calib_img/camera_intrinsic_matrix.txt')
dist = np.loadtxt('calib_img/camera_distortion_matrix.txt')

undistort = Undistort(intrinsics, dist)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_img = undistort.undistort(frame, with_cuda = True)
    cv2.imshow("Undistorted", undistorted_img)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
