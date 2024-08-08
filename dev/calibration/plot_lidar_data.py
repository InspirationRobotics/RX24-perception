import numpy as np
import matplotlib.pyplot as plt

# Load the lidar data
path = 'dev/calibration/calib_img_temp_20240801-155919/lidar_5.npy'
lidar_data = np.load(path, allow_pickle=True)

# Plot the lidar data
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.autoscale(enable=False)

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

# Plot the lidar data
for point_cloud in lidar_data:
    pc = point_cloud[::5] # Downsample the point cloud (grabs every 5th point)
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1)

plt.show(block=True)
plt.ioff()
