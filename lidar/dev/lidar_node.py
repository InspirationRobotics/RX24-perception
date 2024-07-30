import time
import numpy as np
from typing import List
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from lidar_core import Lidar, CustomPointCloud, PointCloud2

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.lidar = Lidar('lidar', decay_rate=0.2)
        self.lidar.add_callback(self.lidar_callback)
        self.subcription = self.create_subscription(PointCloud2, self.lidar.topic, self.lidar.lidar_callback, 10)
        
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def setup_plot(self):
        self.ax.clear()
        # Set contant dimensions
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-2, 2)
        # Set labels
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_title('Lidar data')
        self.ax.autoscale(enable=False)

    def lidar_callback(self, buffer : List[CustomPointCloud]):
        self.setup_plot()
        for point_cloud in buffer:
            self.ax.scatter(point_cloud.points[:,0], point_cloud.points[:,1], point_cloud.points[:,2], s=1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()