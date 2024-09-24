import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_transformation

'''
Function to plot 3D points
Parameters:
- points: numpy array of shape (N, 3), where N is the number of 3D points.
- title: A string to set the plot title.
- ax: A matplotlib 3D axis on which the points will be plotted.
- color: A string representing the color of the points in the plot (default is 'b' for blue).
'''
def plot_points(points, title, ax, color='b'):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

'''
Function to select points in a point cloud
Parameters: 
- lidar_data: A numpy array representing the 3D LiDAR point cloud.
Returns: 
- selected_points: A numpy array of shape (N, 3) with user-selected 3D points from the LiDAR point cloud.
'''
def select_points_in_point_cloud(lidar_data):
    selected_points = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(4, 10)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 0)
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

    # Function to handle keyboard events for point selection and movement
    def on_key(event):
        nonlocal current_point, selected_index, current_point_plot
        if event.key == 'enter':
            selected_points.append(current_point.copy())
            selected_index += 1
            print(f"Selected Point {selected_index}: {current_point}")
        elif event.key == 'up':
            current_point[2] += 0.05
        elif event.key == 'down':
            current_point[2] -= 0.05
        elif event.key == 'left':
            current_point[1] -= 0.05
        elif event.key == 'right':
            current_point[1] += 0.05
        elif event.key == '.':
            current_point[0] += 0.05
        elif event.key == ',':
            current_point[0] -= 0.05
        
        current_point_plot.remove()
        current_point_plot = ax.scatter([current_point[0]], [current_point[1]], [current_point[2]], color='r', s=20)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)
    plt.ioff()

    return np.array(selected_points, dtype=np.float32)

'''
Function to select points in an image
This function allows the user to interactively click on an image to select 2D points.
These points are used to compute correspondences with 3D LiDAR points during calibration.
Parameters:
- image: A 2D numpy array representing the camera image.
- window_name: A string representing the name of the window where the image is displayed (default is 'Select Points').
Returns:
- points: A numpy array of shape (N, 2) with user-selected 2D points from the image.
'''
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

'''
Function to overlay a point cloud onto a camera image
This function projects 3D LiDAR points into 2D image space using a transformation matrix and overlays
the projected points onto the original camera image. Points are color-coded based on their depth.
Parameters:
- image: A 2D numpy array representing the camera image.
- point_cloud: A 3D numpy array representing the LiDAR point cloud (can be reshaped if needed).
- selected_points: A numpy array of selected 3D points for alignment.
- camera_matrix: Camera intrinsic matrix used to project the 3D points onto the 2D image.
- rvec: Rotation vector from the transformation matrix.
- tvec: Translation vector from the transformation matrix.
Returns:
- image_copy: A copy of the original image with the projected points overlaid.
'''
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

def print_image_instructions():
    instructions = """
    ============ LiDAR and Camera Calibration Tool ============
    Instructions for Image Point Selection:

    - Select corresponding key points in the image. 
    Remember the order that you select the points in.

    - You should select an even number of points that is greater or equal to 4. 

    - Click to select a point on the image
    - Press 'q' to exit the point selection process.

    ===========================================================
    """
    print(instructions)
    
def print_pc_instructions():
    instructions = """
    ============ LiDAR and Camera Calibration Tool ============
    Instructions for Lidar Point Selection:

    - Select corresponding key points in the point cloud. 
    These selections should be in the same order that the points on the image were selected.

    - Use the arrow keys to move the current point:
    Up Arrow    : Increase Z-coordinate (move upward)
    Down Arrow  : Decrease Z-coordinate (move downward)
    Left Arrow  : Decrease Y-coordinate (move left)
    Right Arrow : Increase Y-coordinate (move right)
    ',' Key     : Decrease X-coordinate (move backward)
    '.' Key     : Increase X-coordinate (move forward)

    - Press Enter to select the current point and save it.
    - Press 'q' to exit the point selection process.

    ===========================================================
    """
    print(instructions)



def main():
    CAMERA_INTRINSIC_MATRIX_FILE = 'camera/dev/calibration/calib_img/camera_intrinsic_matrix.txt'
    camera_matrix = np.loadtxt(CAMERA_INTRINSIC_MATRIX_FILE)

    # Input LiDAR data in numpy file
    lidar_data = np.load("dev/calibration/calib_img_temp_20240910-121022/lidar_6.npy", allow_pickle=True)

    # Input corresponding image file
    image_path = 'dev/calibration/calib_img_temp_20240910-121022/image_6.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    print_image_instructions()
    camera_corners = select_points_in_image(image)
    print("Camera Corners:", camera_corners)

    print_pc_instructions()
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
