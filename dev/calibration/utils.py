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
    
    camera_corners = camera_corners.astype(np.float32)
    lidar_corners = lidar_corners.astype(np.float32)
    
    success, rvec, tvec = cv2.solvePnP(lidar_corners, camera_corners, camera_matrix, dist_coeffs)
    if not success:
        raise ValueError("solvePnP failed to find a valid transformation")
    
    R, _ = cv2.Rodrigues(rvec)
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()
    
    return transformation_matrix

def apply_transformation_to_point_cloud(point_cloud, transformation_matrix):
    original_shape = point_cloud.shape
    point_cloud = point_cloud.reshape(-1, 3)
    ones = np.ones((point_cloud.shape[0], 1))
    point_cloud_homogeneous = np.hstack([point_cloud, ones])
    transformed_points = (transformation_matrix @ point_cloud_homogeneous.T).T
    
    return transformed_points[:, :3].reshape(original_shape)
