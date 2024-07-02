import cv2
import numpy as np

cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Undistorted", 640, 480)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Original", 640, 480)

# load the camera intrinsic matrix and distortion coefficients
intrinsics = np.loadtxt('/home/inspiration/RX24-perception/camera/img/camera_intrinsic_matrix.txt')
dist = np.loadtxt('/home/inspiration/RX24-perception/camera/img/camera_distortion_matrix.txt')

# optimal camera matrix
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (1920, 1080), 1, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break

    # undistort the image
    undistorted_img = cv2.undistort(frame, intrinsics, dist, None, new_camera_matrix)

    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    cv2.imshow("Undistorted", undistorted_img)
    cv2.imshow("Original", frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break