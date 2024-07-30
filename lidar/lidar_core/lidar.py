import time
import numpy as np
from typing import List
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Imu
from sensor_msgs_py import point_cloud2 as pc2

class CustomPointCloud:
    def __init__(self,  header : Header, points : np.ndarray):
        self.timestamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self.local_timestamp = time.time()
        self.points = points

class Lidar:

    def __init__(self, lidar_id : str, decay_rate : float = 0.1, *, transformation_matrix : np.ndarray = np.eye(4)):
        self.lidar_id = lidar_id
        self.decay_rate = decay_rate
        self.transformation_matrix = transformation_matrix
        self.buffer : List[CustomPointCloud] = []

        self.topic = f"/livox/{lidar_id}"
        self.angular_velocity = None
        self.linear_acceleration = None

    def add_callback(self, callback):
        self.callback = callback

    def lidar_callback(self, point_cloud : PointCloud2):
        points_list = []
        cloud : np.ndarray = pc2.read_points(point_cloud, field_names=("x", "y", "z"), skip_nans=True)
        for pt in cloud:
            points_list.append([pt[0], pt[1], pt[2]])

        points = np.array(points_list)
        self.buffer.append(CustomPointCloud(point_cloud.header, points))

        current_time = time.time()
        for i in range(len(self.buffer)-1, -1, -1):
            if current_time - self.buffer[i].local_timestamp > self.decay_rate:
                self.buffer.pop(i)

        if self.callback is not None:
            self.callback(self.buffer)

    def imu_callback(self, imu_msg : Imu):
        self.angular_velocity = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
        self.linear_acceleration = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])

    def get_points(self):
        return self.buffer