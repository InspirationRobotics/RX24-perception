import cv2
import numpy as np
from pathlib import Path

class Undistort:
    def __init__(self, intrinsics : Path | str | np.ndarray, distortion : Path | str | np.ndarray, width : int, height : int):
    
        if isinstance(intrinsics, (Path, str)):
            intrinsics = np.loadtxt(intrinsics)
        if isinstance(distortion, (Path, str)):
            distortion = np.loadtxt(distortion)

        self.intrinsics = intrinsics
        self.dist = distortion
        self.width = width
        self.height = height

        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (width, height), 1, (width, height))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, self.new_camera_matrix, (width, height), 5)

    def undistort(self, frame, crop = True):
        if crop:
            return self.undistort_and_crop(frame)
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]

    def undistort_and_crop(self, frame):
        undistorted_img = self.undistort(frame, False)
        return self.undistort_roi(undistorted_img)