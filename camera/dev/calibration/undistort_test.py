import cv2
import numpy as np

cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL) 

cv2.resizeWindow("Undistorted", 640, 480)
cv2.resizeWindow("Original", 640, 480)

# load the camera intrinsic matrix and distortion coefficients
intrinsics = np.loadtxt('calib_img/camera_intrinsic_matrix.txt')
dist = np.loadtxt('calib_img/camera_distortion_matrix.txt')

# optimal camera matrix
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (1920, 1080), 1, (1920, 1080))
# Mapping function
mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, dist, None, new_camera_matrix, (1920,1080), 5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break
    
    # undistorted_img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    #--- CUDA TESTING ---

    cuMapX = cv2.cuda.GpuMat(mapx)
    cuMapY = cv2.cuda.GpuMat(mapy)
    cuFrame = cv2.cuda.GpuMat(frame)
    undistorted_img = cv2.cuda.remap(cuFrame, cuMapX, cuMapY, cv2.INTER_LINEAR)
    undistorted_img = undistorted_img.download()

    # -----

    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    cv2.imshow("Undistorted", undistorted_img)
    cv2.imshow("Original", frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break