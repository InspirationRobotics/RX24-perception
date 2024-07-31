import numpy as np
import matplotlib.pyplot as plt

# Load the lidar data
lidar_data = np.load("calib_img_temp/lidar_data_1722392174.6908574.npy", allow_pickle=True)

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
    ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s=1)

plt.show(block=True)
plt.ioff()
