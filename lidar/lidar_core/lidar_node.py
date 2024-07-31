import time
import numpy as np
from lidar_core.utils import *
from lidar_core import Lidar, CustomPointCloud

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

        self.active = False
        
    def create_lidar_subscription(self, lidar : Lidar):
        callback_group = MutuallyExclusiveCallbackGroup()
        subscription = self.create_subscription(PointCloud2, lidar.topic, lidar.lidar_callback, 10, callback_group=callback_group)
        self.lidar_subscriptions.append(subscription)

    def init_subscriptions(self):
        for lidar in self.lidars:
            self.create_lidar_subscription(lidar)

    def start(self):
        if self.active:
            return
        self.lidar_executor = MultiThreadedExecutor()
        self.lidar_executor.add_node(self)
        self.spin_thread.start()
        self.active = True
        print("Started lidar node...")

    def stop(self):
        if not self.active:
            return
        self.spin_thread.stop()
        self.lidar_executor.shutdown()
        self.active = False
        print("Stopped lidar node...")

    def _run(self, dummy):
        # This is a blocking call so if the lidar stops publishing, the node will hang
        # For some reason setting the timeout to 0 reduces the rate of the lidar data
        self.lidar_executor.spin_once() 