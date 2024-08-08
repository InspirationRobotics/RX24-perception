import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils import calculate_transformation, apply_transformation_to_point_cloud
# from undistort import Undistort

# Function to generate synthetic LiDAR data points
def generate_filled_rectangle_lidar_data(width=2.0, height=1.0, num_points_per_unit_area=25, noise_level=0.01):
    x = np.linspace(-width / 2, width / 2, int(width * num_points_per_unit_area))
    y = np.linspace(-height / 2, height / 2, int(height * num_points_per_unit_area))
    xv, yv = np.meshgrid(x, y)
    points = np.vstack((xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten()))).T

    # Add noise to the points to simulate real-world data
    noise = np.random.normal(0, noise_level, points.shape)
    points += noise

    return points

def detect_rectangle_corners(point_cloud):
    # Find the minimum and maximum coordinates in each dimension
    min_x, min_y, min_z = np.min(point_cloud, axis=0)
    max_x, max_y, max_z = np.max(point_cloud, axis=0)
    
    # Define the corners of the rectangle based on the min and max coordinates
    corners = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z]
    ])
    
    return corners

# Function to plot 3D points
def plot_points(points, title, ax, color='b'):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def select_points(image, window_name='Select Points'):
    points = []
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, image)
    
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(points, dtype=np.float32)

def visualize_point_cloud(points, title="Point Cloud"):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()

def calculate_reprojection_error(camera_corners, lidar_corners, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(lidar_corners, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    error = np.sqrt(np.mean(np.sum((camera_corners - projected_points) ** 2, axis=1))) # Euclidean Distance
    return error

def main():
    # File paths for camera intrinsic and distortion matrices
    CAMERA_INTRINSIC_MATRIX_FILE = 'camera/dev/calibration/calib_img/camera_intrinsic_matrix.txt'
    CAMERA_DISTORTION_MATRIX_FILE = 'camera/dev/calibration/calib_img/camera_distortion_matrix.txt'

    # Load camera intrinsic and distortion matrices from files
    camera_matrix = np.loadtxt(CAMERA_INTRINSIC_MATRIX_FILE)
    dist_coeffs = np.loadtxt(CAMERA_DISTORTION_MATRIX_FILE)
    
    # Initialize undistortion object with camera parameters
    # undistorter = Undistort(camera_matrix, dist_coeffs, 1920, 1080)
    
    # Generate synthetic LiDAR data
    synthetic_lidar_data = generate_filled_rectangle_lidar_data()
    visualize_point_cloud(synthetic_lidar_data, "Original LiDAR Points")
    print(synthetic_lidar_data.shape)

    # Automatically detect corners in the LiDAR point cloud
    lidar_corners = detect_rectangle_corners(synthetic_lidar_data)
    
    # Read and undistort an image using the camera parameters
    image = cv2.imread('camera/dev/calibration/calib_img/image_7.jpg')
    # undistorted_image = undistorter.undistort(image)

    # DEBUG STATEMENT
    if image is None:
        # print(f"Error: Unable to load image at {image_path}")
        return
    
    # Manually select key points in the camera image
    print("Select corresponding key points in the camera image.")
    camera_corners = select_points(image)
    
    # DEBUG STATEMENTS
    print("Camera Corners:", camera_corners)
    print("LiDAR Corners:", lidar_corners)
    print(f"Number of Camera Corners: {len(camera_corners)}")
    print(f"Number of LiDAR Corners: {len(lidar_corners)}")
    
    if len(camera_corners) != len(lidar_corners):
        print("Error: The number of detected corners in the camera image and the LiDAR point cloud do not match.")
        return
    
    # Calculate the transformation matrix between camera and LiDAR coordinates
    transformation_matrix = calculate_transformation(camera_corners, lidar_corners, camera_matrix, dist_coeffs)

    # Apply the transformation to the synthetic LiDAR data
    transformed_point_cloud = apply_transformation_to_point_cloud(synthetic_lidar_data, transformation_matrix)
    visualize_point_cloud(transformed_point_cloud, "Transformed LiDAR Points")

    # Plot the original and transformed LiDAR points
    fig = plt.figure()

    # Plot original LiDAR points
    ax = fig.add_subplot(121, projection='3d')
    plot_points(synthetic_lidar_data, 'Original LiDAR Points', ax, color='b')
    
    # Plot transformed LiDAR points
    ax = fig.add_subplot(122, projection='3d')
    plot_points(transformed_point_cloud, 'Transformed LiDAR Points', ax, color='r')
    plt.show()
    
    print("Transformation Matrix:\n", transformation_matrix)

    rvec, _ = cv2.Rodrigues(transformation_matrix[:3, :3])
    tvec = transformation_matrix[:3, 3]
    reprojection_error = calculate_reprojection_error(camera_corners, lidar_corners, rvec, tvec, camera_matrix, dist_coeffs)
    print(f"Reprojection Error: {reprojection_error}")

if __name__ == "__main__":
    main()
