import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_transformation

# Function to plot 3D points
def plot_points(points, title, ax, color='b'):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def select_points_in_point_cloud(lidar_data):
    selected_points = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(1, 3)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-0.5, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Select Points in Point Cloud')
    
    for point_cloud in lidar_data:
        pc = point_cloud[::5]  # Downsample the point cloud (grabs every 5th point)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    
    plt.ion()

    current_point = [0, 0, 0]
    selected_index = 0
    current_point_plot = ax.scatter([current_point[0]], [current_point[1]], [current_point[2]], color='r', s=20)

    def on_key(event):
        nonlocal current_point, selected_index, current_point_plot
        if event.key == 'enter':
            selected_points.append(current_point.copy())
            selected_index += 1
            print(f"Selected Point {selected_index}: {current_point}")
        elif event.key == 'up':
            current_point[2] += 0.015
        elif event.key == 'down':
            current_point[2] -= 0.015
        elif event.key == 'left':
            current_point[1] -= 0.015
        elif event.key == 'right':
            current_point[1] += 0.015
        elif event.key == '.':
            current_point[0] += 0.015
        elif event.key == ',':
            current_point[0] -= 0.015
        
        current_point_plot.remove()
        current_point_plot = ax.scatter([current_point[0]], [current_point[1]], [current_point[2]], color='r', s=20)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)
    plt.ioff()

    return np.array(selected_points, dtype=np.float32)

def select_points_in_image(image, window_name='Select Points'):
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)

    if points.ndim == 2:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    else:
        for point_cloud in points:
            pc = point_cloud[::5]
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    
    plt.show()

def overlay_point_cloud_on_image(image, point_cloud, selected_points, camera_matrix, rvec, tvec):
    if point_cloud.ndim == 3:
        point_cloud = point_cloud.reshape(-1, 3)

    if point_cloud.dtype != np.float32:
        point_cloud = point_cloud.astype(np.float32)

    projected_points, _ = cv2.projectPoints(point_cloud, rvec, tvec, camera_matrix, None)

    depths = point_cloud[:, 2]
    norm_depths = cv2.normalize(depths, None, 0, 255, cv2.NORM_MINMAX)
    colors = cv2.applyColorMap(norm_depths.astype(np.uint8), cv2.COLORMAP_JET)

    image_copy = image.copy()
    for point, color in zip(projected_points, colors):
        x, y = int(point[0][0]), int(point[0][1])

        if 0 <= x < image_copy.shape[1] and 0 <= y < image_copy.shape[0]:
            color = tuple(map(int, color[0]))
            cv2.circle(image_copy, (x, y), 5, color, -1)
    
    return image_copy


def main():
    CAMERA_INTRINSIC_MATRIX_FILE = 'camera/dev/calibration/calib_img/camera_intrinsic_matrix.txt'
    camera_matrix = np.loadtxt(CAMERA_INTRINSIC_MATRIX_FILE)
    print("Camera Matrix:\n", camera_matrix)

    lidar_data = np.load("dev/calibration/calib_img_temp_20240829-190725/lidar_4.npy", allow_pickle=True)
    print("LiDAR Data Shape:", lidar_data.shape)

    image_path = 'dev/calibration/calib_img_temp_20240829-190725/image_4.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    print("Select corresponding key points in the camera image.")
    camera_corners = select_points_in_image(image)
    print("Camera Corners:", camera_corners)

    print("Select corresponding key points in the point cloud.")
    lidar_corners = select_points_in_point_cloud(lidar_data)
    print("LiDAR Corners:", lidar_corners)

    if len(camera_corners) != len(lidar_corners):
        print("Error: The number of detected corners in the camera image and the LiDAR point cloud do not match.")
        return

    transformation_matrix = calculate_transformation(camera_corners, lidar_corners, camera_matrix, None)
    print(f"Transformation Matrix: \n {transformation_matrix}")

    rvec, _ = cv2.Rodrigues(transformation_matrix[:3, :3])
    tvec = transformation_matrix[:3, 3]
    overlayed_image = overlay_point_cloud_on_image(image, lidar_data, lidar_corners, camera_matrix, rvec, tvec)
    cv2.imshow("Overlayed Image", overlayed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
