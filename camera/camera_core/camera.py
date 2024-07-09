import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from threading import Thread, Lock
from camera_core.undistort import Undistort
from camera_core.ml_model import ML_Model
import time

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

    def __init__(self, /, camera_id : int = 0, *, model : ML_Model = None, resolution : Tuple[int, int] = (1920, 1080), fps : int = 30, video_path : str | Path = None, camera_type : str = 'wide'):
        self.camera_path = f'/dev/video{camera_id}'
        self.video_path = video_path
        self.resolution = resolution
        self.fps = fps
        self.model = model
        self.load_calibration(camera_type)
        self.stream = False
        self.done_init = False
        self.raw_frame = None
        self.frame = None
        self.results = []
        self.camera_lock = Lock()
        self.model_lock = Lock()
        
    def load_calibration(self, camera_type : str):
        pre_path = Path('/home/inspiration/RX24-perception/camera/camera_core/config') # TODO Make this relative
        dist_calibration_path = pre_path / Path(f'{camera_type}/camera_distortion_matrix.txt')
        int_calibration_path = pre_path / Path(f'{camera_type}/camera_intrinsic_matrix.txt')
        self.undistort = Undistort(int_calibration_path, dist_calibration_path, self.resolution)

    def load_model(self, model_object : ML_Model):
        with self.model_lock:
            self.model = model_object

    def load_model(self, model_path : str | Path, model_type : str = "YOLO", *, half_precision: bool = False):
        with self.model_lock:
            self.model = ML_Model(model_path, model_type, half_precision=half_precision)

    def switch_model(self, model_object : ML_Model):
        with self.model_lock:
            self.model = model_object

    def switch_model(self, model_path : str | Path, model_type: str = "YOLO", *, half_precision: bool = False):
        with self.model_lock:
            self.model.switch_model(model_path, model_type, half_precision=half_precision)

    def start_model(self):
        if self.stream == False:
            print("Error: Camera stream not started, cannot start model thread")
            return
        self.run_model = True
        self.model_thread = Thread(target=self.model_background_thread)
        self.model_thread.start()
        print("Model thread started")

    def stop_model(self):
        self.run_model = False
        self.model_thread.join()
        print("Model thread stopped")

    def start_stream(self):
        self.stream = True
        self.camera_thread = Thread(target=self.camera_background_thread)
        self.camera_thread.start()
        print("Camera stream started")

    def stop_stream(self):
        self.stream = False
        self.done_init = False
        self.camera_thread.join()
        print("Camera stream stopped")

    def get_size(self, undistort = True) -> Tuple[int, int]:
        if undistort:
            return self.undistort.get_roi_dimensions()
        return self.resolution
    
    def warmup(self, frame : np.ndarray = None) -> np.ndarray:
        temp_frame = self.warmup_undistort(frame)
        self.model.predict(temp_frame)

    def warmup_undistort(self, frame : np.ndarray = None) -> np.ndarray:
        if frame is None:
            # Make a frame with random values
            warmup_frame = np.random.randint(0, 255, (self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        else:
            warmup_frame = frame
        warmup_frame = self.undistort.undistort(warmup_frame, with_cuda=True)
        print("Camera undistortion warmed up...")
        return warmup_frame

    def get_latest_frame(self, *, undistort = False, with_cuda = False) -> Image:
        with self.camera_lock:
            if not self.done_init:
                return None
            # If there is a raw frame use that and then reset it. 
            # Otherwise check if there is a processed frame and use that. If not return None
            if self.raw_frame is not None:
                frame = self.raw_frame
                self.raw_frame = None
            elif self.frame is not None:
                frame = self.frame
            else:
                return None
        if undistort:
            frame = self.undistort.undistort(frame, with_cuda = with_cuda)
        self.frame = frame
        return Image(frame)
    
    def get_latest_model_results(self) -> List:
        with self.model_lock:
            return self.results
    
    def model_background_thread(self):
        while self.run_model:
            with self.camera_lock:
                frame = self.frame
            if self.stream:
                with self.model_lock:
                    if self.model is not None:
                        self.results = self.model.predict(frame)
            time.sleep(1/(self.fps))

    def camera_background_thread(self):
        if self.video_path == None:
            capture_flag = f'v4l2src device={self.camera_path} ! image/jpeg, width={self.resolution[0]}, height={self.resolution[1]}, framerate={self.fps}/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink'
            try:
                self.cap = cv2.VideoCapture(capture_flag)
            except:
                print("Error: failed to open camera using GStreamer; Defaulting to OpenCV")
                self.cap = cv2.VideoCapture(self.camera_path)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        self.done_init = True
        while self.stream:
            with self.camera_lock:
                ret, self.raw_frame = self.cap.read()
                if not ret:
                    print("Error: failed to capture image")
                    self.stream = False
                    break
            time.sleep(1/(self.fps))

        self.cap.release()
        return
        
