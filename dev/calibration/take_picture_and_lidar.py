import os
import cv2
import time
import rclpy
import numpy as np
from pathlib import Path
from camera_core import Camera, Image
from lidar_core import Lidar, LidarNode

# Get current data and format as string
time_stamp = time.strftime("%Y%m%d-%H%M%S")
# Set maximum number of images to save
MAX_IMAGES = 10
# Set path to save images
CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / f"calib_img_temp_{time_stamp}"

if not IMG_FILE_PATH.exists():
    os.mkdir(IMG_FILE_PATH)

rclpy.init(args=None)

lidar = Lidar('lidar', decay_rate=0.2)
lidar_node = LidarNode([lidar])

camera = Camera(bus_addr=[1,7], camera_type='port')
camera.warmup()

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Camera", 640, 480)

lidar_node.start()
camera.start()

i=0
while True:
    frame = camera.get_latest_frame(undistort=True, with_cuda=True)
    if frame is None:
        continue
    frame = frame.frame
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
        file_name = f"lidar_{i}.npy"
        np.save(str(IMG_FILE_PATH / f"{file_name}"), lidar.get_points_np())
        print(f'saving lidar data: {file_name}')

    if i >= MAX_IMAGES:
        print("Max number of images saved")
        break

if i == 0:
    os.removedirs(IMG_FILE_PATH)

camera.stop()
lidar_node.stop()
cv2.destroyAllWindows()
rclpy.shutdown()