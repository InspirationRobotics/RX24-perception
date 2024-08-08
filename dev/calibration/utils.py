import numpy as np
import cv2

def calculate_transformation(camera_corners, lidar_corners, camera_matrix, dist_coeffs):
    """
    Calculate the transformation matrix using the camera and LiDAR points.
    
    :param camera_corners: (N, 2) array of camera corner points in image space (2D).
    :param lidar_corners: (N, 3) array of LiDAR corner points in world space (3D).
    :param camera_matrix: Camera intrinsic matrix.
    :param dist_coeffs: Camera distortion coefficients.
    :return: (4, 4) transformation matrix.
    """
    assert len(camera_corners) == len(lidar_corners), "Number of camera and LiDAR points must be the same"
    
    # Convert camera corners to homogeneous coordinates
    camera_corners_h = np.hstack([camera_corners, np.ones((camera_corners.shape[0], 1))])
    
    # Estimate the pose using solvePnP
    success, rvec, tvec = cv2.solvePnP(lidar_corners, camera_corners, camera_matrix, dist_coeffs)
    if not success:
        raise ValueError("solvePnP failed to find a valid transformation")
    
    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()
    
    return transformation_matrix

def apply_transformation_to_point_cloud(point_cloud, transformation_matrix):
    """
    Apply the transformation matrix to the point cloud.
    
    :param point_cloud: (N, 3) array of points.
    :param transformation_matrix: (4, 4) transformation matrix.
    :return: Transformed point cloud (N, 3).
    """
    original_shape = point_cloud.shape
    point_cloud = point_cloud.reshape(-1, 3)  # Flatten the point cloud array to 2D
    ones = np.ones((point_cloud.shape[0], 1))
    point_cloud_homogeneous = np.hstack([point_cloud, ones])
    transformed_points = (transformation_matrix @ point_cloud_homogeneous.T).T
    
    return transformed_points[:, :3].reshape(original_shape)  # Reshape back to the original shape if needed