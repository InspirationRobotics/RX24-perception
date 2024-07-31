import cv2
import numpy as np
import os
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from lidar_core import Lidar, CustomPointCloud, PointCloud2
from camera_core import Camera, Image

CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / "calib_img_temp"

if not IMG_FILE_PATH.exists():
    os.mkdir(IMG_FILE_PATH)

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.lidar = Lidar('lidar', decay_rate=0.2)
        self.subcription = self.create_subscription(PointCloud2, self.lidar.topic, self.lidar.lidar_callback, 10)

    def save_lidar_data(self, timestamp):
        np.save(str(IMG_FILE_PATH / f"lidar_data_{time_stamp}.npy"), self.lidar.get_points_np())


rclpy.init(args=None)
node = LidarNode()

camera = Camera(4)
camera.warmup()

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Camera", 640, 480)

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
        time_stamp = time.time()
        file_name = f"image_{i}_{time_stamp}.jpg"
        print(f'saving image: {file_name}')
        cv2.imwrite(str(IMG_FILE_PATH / file_name), frame)
        node.save_lidar_data(time_stamp)
        print(f'saving lidar data: lidar_data_{time_stamp}.npy')

    if i == 5:
        print("Max number of images saved")
        break
    rclpy.spin_once(node)
    time.sleep(1/35)

camera.stop()
cv2.destroyAllWindows()
rclpy.shutdown()