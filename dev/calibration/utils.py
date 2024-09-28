import numpy as np
import cv2

"""
Calculate the transformation matrix using the camera and LiDAR points.
Parameters: 
- camera_corners: (N, 2) array of camera corner points in image space (2D).
- lidar_corners: (N, 3) array of LiDAR corner points in world space (3D).
- camera_matrix: Camera intrinsic matrix.
- dist_coeffs: Camera distortion coefficients.
Returns: 
- (4, 4) transformation matrix.
"""
def calculate_transformation(camera_corners, lidar_corners, camera_matrix, dist_coeffs):
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