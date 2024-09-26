import numpy as np
import matplotlib.pyplot as plt

# Load the lidar data
path = '/home/eesh/RX24/RX24-perception/dev/calibration/calib_img_temp_20240905-115452/lidar_5.npy'
lidar_data = np.load(path, allow_pickle=True)

# Plot the lidar data
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.autoscale(enable=True)

# Set contant dimensions
ax.set_xlim(-1, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-2, 2)

# Set labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Set title
ax.set_title('Lidar data')

# Remove all points above the z=0 plane and below the z=-1.5 plane
def remove_points(points):
    return points[(points[:,2] < 0) & (points[:,2] > -1.5)]



# Plot the lidar data
for point_cloud in lidar_data:
    print(max(point_cloud[:,0]))
    point_cloud = remove_points(point_cloud)
    pc = point_cloud#[::5] # Downsample the point cloud (grabs every 5th point)
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1)

plt.show(block=True)
plt.ioff()
