import cv2
import numpy as np
from lidar_core import OccupancyGrid


SAMPLE_LIDAR_DATA = "/home/eesh/RX24/RX24-perception/dev/calibration/calib_img_temp_20240910-121022/lidar_2.npy"
lidar_data = np.load(SAMPLE_LIDAR_DATA, allow_pickle=True)
SAMPLE_LIDAR_DATA2 = "/home/eesh/RX24/RX24-perception/dev/calibration/calib_img_temp_20240910-121022/lidar_3.npy"
lidar_data2 = np.load(SAMPLE_LIDAR_DATA2, allow_pickle=True)

def test_occ():
    og = OccupancyGrid(37.7749, -122.4194, cell_size=0.2)
    og.update_grid(37.7749, -122.4194, 0, lidar_data)
    og.visualize()
    og.update_grid(37.7749, -122.4194, 2, lidar_data2)
    og.visualize()

if __name__ == "__main__":
    test_occ()