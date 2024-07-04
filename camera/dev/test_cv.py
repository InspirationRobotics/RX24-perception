import cv2

cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink ")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Camera", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break

    cv2.imshow("Camera", frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break