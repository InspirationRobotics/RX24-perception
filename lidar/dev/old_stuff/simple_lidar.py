import rclpy
import time
import numpy as np
import matplotlib.pyplot as plt
from lidar_core import Lidar, LidarNode


def plot_lidar():
    # This is honestly a really messy function like it looks so ugly but it works
    ax.clear()
    # Set contant dimensions
    ax.set_xlim(-1, 5)
    ax.set_ylim(-5, 5)
    # self.ax.set_zlim(-2, 2)
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # self.ax.set_zlabel('z')
    ax.set_title('Lidar data')
    ax.autoscale(enable=False)

    data = lidar.get_points()
    start_time = time.time()
    for point_cloud in data:
        ax.scatter(point_cloud.points[:,0], point_cloud.points[:,1], s=1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("Draw time: ", time.time() - start_time)
    print(f'Point cloud len: {len(data)}')

if __name__ == '__main__':
    # matplotlib config
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.show(block=False)
    # This inits ros2
    rclpy.init(args=None)
    # Creates a lidar object
    lidar = Lidar('lidar', decay_rate=0.2)
    # Adds the object to the node (you can add multiple if you want for multiple lidars)
    lidar_node = LidarNode([lidar])
    # Starts the node
    lidar_node.start()
    # Run the node for 30 seconds
    duration = 30
    start_time = time.time()
    while time.time() - start_time < duration:
        # This function will plot the lidar data, notice how it can grab the latest data at any moment
        # from any lidar added to the node
        plot_lidar()
        time.sleep(1/30)
    
    # Stops the node
    lidar_node.stop()
    # Shutdown ros2
    rclpy.shutdown()