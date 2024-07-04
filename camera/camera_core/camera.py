import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from threading import Thread
from camera_core.undistort import Undistort

'''
This is the main camera class that will be used to interact with the camera.
It will handle the camera undistortion and getting the latest frame.
The Camera object is imported into your code and used to get the latest frame.
'''

class Image:
    def __init__(self, frame : np.ndarray):
        self.frame = frame
        self.dimensions = (frame.shape[1], frame.shape[0])

class Camera:

    def __init__(self, /, camera_id : int = 0, *, resolution : Tuple[int, int] = (1920, 1080), fps : int = 30, camera_type : str = 'wide'):
        self.camera_path = f'/dev/video{camera_id}'
        self.resolution = resolution
        self.fps = fps
        self.load_calibration(camera_type)
        self.stream = False
        self.frame = None
        
    def load_calibration(self, camera_type : str):
        dist_calibration_path = Path(f'config/{camera_type}/camera_distortion_matrix.txt')
        int_calibration_path = Path(f'config/{camera_type}/camera_intrinsic_matrix.txt')
        self.undistort = Undistort(int_calibration_path, dist_calibration_path, self.resolution)

    def start_stream(self):
        self.stream = True
        self.camera_thread = Thread(target=self.camera_background_thread)
        self.camera_thread.start()
        print("Camera stream started")

    def stop_stream(self):
        self.stream = False
        self.camera_thread.join()
        print("Camera stream stopped")

    def get_size(self, undistort = True) -> Tuple[int, int]:
        if undistort:
            return self.undistort.get_roi_dimensions()
        return self.resolution

    def get_latest_frame(self, *, undistort = True, with_cuda = False) -> Image:
        if self.frame is None:
            return None
        if undistort:
            return Image(self.undistort.undistort(self.frame, with_cuda = with_cuda))
        return Image(self.frame)

    def camera_background_thread(self):
        capture_flag = f'v4l2src device={self.camera_path} ! image/jpeg, width={self.resolution[0]}, height={self.resolution[1]}, framerate={self.fps}/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink'
        try:
            self.cap = cv2.VideoCapture(capture_flag)
        except:
            print("Error: failed to open camera using GStreamer; Defaulting to OpenCV")
            self.cap = cv2.VideoCapture(self.camera_path)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        while self.stream:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: failed to capture image")
                break
            self.frame = frame
        
