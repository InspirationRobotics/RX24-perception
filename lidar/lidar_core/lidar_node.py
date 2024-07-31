import time
import numpy as np
from .utils import *
from .lidar import Lidar, CustomPointCloud

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import PointCloud2, PointField, Imu

class LidarNode(Node):

    def __init__(self, lidars : List[Lidar]):
        super().__init__('lidar_node')
        self.lidars = lidars
        self.lidar_subscriptions = []
        self.init_subscriptions()
        self.spin_thread = BackgroundThread(self._run, rate = 30)

    def create_lidar_subscription(self, lidar : Lidar):
        callback_group = MutuallyExclusiveCallbackGroup()
        subscription = self.create_subscription(PointCloud2, lidar.topic, lidar.lidar_callback, 10, callback_group=callback_group)
        self.lidar_subscriptions.append(subscription)

    def init_subscriptions(self):
        for lidar in self.lidars:
            self.create_lidar_subscription(lidar)

    def start(self):
        self.lidar_executor = MultiThreadedExecutor()
        self.lidar_executor.add_node(self)
        self.spin_thread.start()

    def stop(self):
        self.spin_thread.stop()
        self.lidar_executor.shutdown()

    def _run(self, dummy):
        self.lidar_executor.spin_once()