import cv2
import time
import random
import numpy as np
from lidar_core import OccupancyGrid


SAMPLE_LIDAR_DATA = "/home/eesh/RX24/RX24-perception/dev/calibration/calib_img_temp_20240910-121022/lidar_2.npy"
lidar_data = np.load(SAMPLE_LIDAR_DATA, allow_pickle=True)
SAMPLE_LIDAR_DATA2 = "/home/eesh/RX24/RX24-perception/dev/calibration/calib_img_temp_20240910-121022/lidar_3.npy"
lidar_data2 = np.load(SAMPLE_LIDAR_DATA2, allow_pickle=True)

def test_occ():
    og = OccupancyGrid(37.7749, -122.4194, cell_size=0.2)
    starttime = time.time()
    og.update_grid(37.7749, -122.4194, 0, lidar_data)
    print("Time taken for update_grid: ", time.time() - starttime)
    og.visualize(show=True)
    starttime = time.time()
    og.update_grid(37.7749, -122.4194, 90, lidar_data)
    print("Time taken for update_grid: ", time.time() - starttime)
    og.visualize(show=True)

def test_live_occ():
    cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
    og = OccupancyGrid(37.7749, -122.4194, cell_size=0.2)
    prev_degree = 0
    prev_lat = 37.7749
    prev_lon = -122.4194
    while True:
        #random degree within 3 degrees of previous degree
        degree = random.randint(prev_degree-10, prev_degree+10)
        # random lat within 0.0001 of previous lat
        lat = prev_lat + random.uniform(-0.000001, 0.000001)
        # random lon within 0.0001 of previous lon
        lon = prev_lon + random.uniform(-0.000001, 0.000001)
        # random lidar data (either 1 or 2)
        lidar = random.choice([lidar_data, lidar_data2])
        starttime = time.time()
        og.update_grid(lat, lon, degree, lidar)
        print("Time taken for update_grid: ", time.time() - starttime)
        frame = og.visualize(show=False)
        frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Occupancy Grid", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_degree = degree
        prev_lat = lat
        prev_lon = lon
        # time.sleep(0.1)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # test_occ()
    test_live_occ()