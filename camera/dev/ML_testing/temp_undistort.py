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

    def undistort_only(self, frame, *, cuda = False):
        if cuda:
            cuMapX = cv2.cuda.GpuMat(self.mapx)
            cuMapY = cv2.cuda.GpuMat(self.mapy)
            cuFrame = cv2.cuda.GpuMat(frame)
            undistorted_img = cv2.cuda.remap(cuFrame, cuMapX, cuMapY, cv2.INTER_LINEAR)
            return undistorted_img.download()
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]

    def undistort(self, frame, *, with_cuda = False):
        undistorted_img = self.undistort_only(frame, cuda = with_cuda)
        return self.undistort_roi(undistorted_img)