import time
import numpy as np
from lidar_core.utils import *

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Imu
from sensor_msgs_py import point_cloud2 as pc2

class CustomPointCloud:
    def __init__(self,  header : Header, points : np.ndarray = None):
        if points is None:
            self._copy(header)
            return
        self.timestamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self.local_timestamp = time.time()
        self.points = points

    def _copy(self, to_copy : 'CustomPointCloud'):
        self.timestamp = to_copy.timestamp
        self.local_timestamp = to_copy.local_timestamp
        self.points = to_copy.points.copy()

class Lidar:

    def __init__(self, lidar_id : str, decay_rate : float = 0.1, *, transformation_matrix : np.ndarray = np.eye(4)):
        self.lidar_id = lidar_id
        self.decay_rate = decay_rate
        self.transformation_matrix = transformation_matrix
        self.buffer : List[CustomPointCloud] = []
        self.buffer_lock = Lock()

        self.topic = f"/livox/{lidar_id}"
        self.angular_velocity = None
        self.linear_acceleration = None
        self.callback = None

    def __str__(self):
        return f"{self.lidar_id}"

    def add_callback(self, callback):
        self.callback = callback

    def process_lidar(self, point_cloud : PointCloud2):
        # Currently takes ~0.02 seconds
        points_list = []
        cloud : np.ndarray = pc2.read_points(point_cloud, field_names=("x", "y", "z"), skip_nans=True)
        for pt in cloud:
            points_list.append([pt[0], pt[1], pt[2]])

        points = np.array(points_list)
        with self.buffer_lock:
            self.buffer.append(CustomPointCloud(point_cloud.header, points))

        if len(self.buffer) > 5:
            current_time = self.buffer[-1].timestamp
            for i in range(len(self.buffer)):
                if current_time - self.buffer[i].timestamp < self.decay_rate:
                    last_valid = i
                    break
            if len(self.buffer) - last_valid < 5:
                last_valid = len(self.buffer) - 5
            self.buffer = self.buffer[last_valid:]

    def lidar_callback(self, point_cloud : PointCloud2):
        self.process_lidar(point_cloud)
        # Call the external callback function
        if self.callback is not None:
            self.callback(self.get_points_np())

    def imu_callback(self, imu_msg : Imu):
        self.angular_velocity = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
        self.linear_acceleration = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])

    def get_points(self) -> List[CustomPointCloud]:
        # Takes ~0.00005 seconds
        list_copy = [] 
        with self.buffer_lock:
            for point_cloud in self.buffer:
                list_copy.append(CustomPointCloud(point_cloud))
        return list_copy
    
    def get_points_np(self):
        with self.buffer_lock:
            return np.array([point_cloud.points for point_cloud in self.buffer])