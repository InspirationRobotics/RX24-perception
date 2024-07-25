import time
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class Lidar(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/combined_lidar',
            self.lidar_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.buffer = []

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Lidar data')
        self.ax.autoscale(enable=False)
        # self.ax.set_aspect('equal')

    def lidar_callback(self, msg : PointCloud2):
        # Make an array of the data in the point cloud
        points_list = []
        cloud : np.ndarray = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        for pt in cloud:
            points_list.append([pt[0], pt[1], pt[2]])

        # Convert the list to a numpy array
        points = np.array(points_list)

        # Add to the buffer and remove data older than 0.2 seconds
        self.buffer.append((time.time(), points))

        # Traverse backwards through the buffer and remove old data
        current_time = time.time()
        for i in range(len(self.buffer)-1, -1, -1):
            if current_time - self.buffer[i][0] > 0.2:
                self.buffer.pop(i)

        # # Plot just the x against y
        self.ax.clear()
        for i in range(len(self.buffer)):
            points = self.buffer[i][1]
            self.ax.plot(points[:, 0], points[:, 1], 'b.')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

def main(args=None):
    rclpy.init(args=args)

    lidar = Lidar()

    rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
