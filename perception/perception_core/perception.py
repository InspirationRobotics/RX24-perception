import cv2

from threading import Thread, Lock

from typing import List, Tuple, Dict
from camera_core import Camera, Image, Results

'''
This class will be imported by the mission handler.
The perception class will be responsible for handling the camera feeds and processing them.

It should have the capability to: (for all cameras or individual cameras)
- Start and stop the cameras
- Get the latest frame
- Record the camera feed
- Load a model and run inference on the camera feed
- Return the inference results (along with the camera it came from)

- There should be a single function which will parse the relevant commands from the mission handler and execute the appropriate functions.
     - It will input a dictionary with the following commands:
        - "start" : [camera_name]
        - "stop" : [camera_name]
        - "record" : [camera_name]
        - "stop_record" : [camera_name]
        - "load_model" : [(camera_name, model_path)]
        - "stop_model" : [camera_name]
     - A camera_name of "all" will apply the command to all cameras.

 - There will also be a dictionary of latest frames and inference results which will be updated by the perception class.
 - The keys for the dictionary will be the camera names and will either have data or be None.
 - There will be a function to get the above dictionary.
'''

class CameraData:

    def __init__(self, image : Image, results : Results):
        self.frame = image.frame
        self.results = results

class Perception:

    camera_addrs = {
        "port": [1,8],
        "center": [1,9],
        "right": [1,10]
        }

    def __init__(self):
        self.active_cameras : Dict[str, Camera] = {}
        self.active_writers : Dict[str, cv2.VideoWriter] = {}

        self.latest_data : Dict[str, CameraData] = {}

        self.active = True
        self.change_lock = Lock()
        self.perception_thread = Thread(target=self.__perception_loop)
        self.perception_thread.start()

    def __handle_all(func):
        def wrapper(self : Perception, camera_names : list):
            with self.change_lock:
                if camera_names is None:
                    return
                if isinstance(camera_names[0], tuple):
                    # Means that the camera_names are in the form of (camera_name, model_path)
                    if "all" == camera_names[0][0]:
                        model_path = camera_names[0][1]
                        camera_names = [(key, model_path) for key in self.camera_addrs.keys()]
                else:
                    if "all" in camera_names:
                        camera_names = list(self.camera_addrs.keys())
                return func(self, camera_names)
        return wrapper

    def __del__(self):
        self.active = False
        self.perception_thread.join(2)
        for camera in self.active_cameras.values():
            camera.stop()

    def __start_camera(self, camera_name : str):
        if camera_name in self.active_cameras:
            return
        camera = Camera(bus_addr=self.camera_addrs[camera_name], camera_type=camera_name)
        camera.start()
        self.active_cameras[camera_name] = camera

    def __stop_camera(self, camera_name : str):
        if camera_name not in self.active_cameras:
            return
        camera = self.active_cameras.pop(camera_name)
        if camera_name in self.active_writers:
            writer = self.active_writers.pop(camera_name)
            writer.release()
        camera.stop()

    def __record_camera(self, camera_name : str):
        if camera_name not in self.active_cameras:
            return
        if camera_name in self.active_writers:
            return
        camera = self.active_cameras[camera_name]
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        writer = cv2.VideoWriter(f'videos/{camera_name}.avi', fourcc, 20.0, camera.get_size())
        self.active_writers[camera_name] = writer

    def __stop_record_camera(self, camera_name : str):
        if camera_name not in self.active_cameras:
            return
        if camera_name not in self.active_writers:
            return
        writer = self.active_writers.pop(camera_name)
        writer.release()

    def __load_model(self, camera_name : str, model_path : str):
        if camera_name not in self.active_cameras:
            return
        camera = self.active_cameras[camera_name]
        camera.switch_model(model_path)
        camera.start_model()

    def __stop_model(self, camera_name : str):
        if camera_name not in self.active_cameras:
            return
        camera = self.active_cameras[camera_name]
        camera.stop_model()

    @__handle_all
    def _start_cameras(self, camera_names : list):
        for camera_name in camera_names:
            self.__start_camera(camera_name)

    @__handle_all
    def _stop_cameras(self, camera_names : list):
        for camera_name in camera_names:
            self.__stop_camera(camera_name)

    @__handle_all
    def _record_cameras(self, camera_names : list):
        for camera_name in camera_names:
            self.__record_camera(camera_name)
    
    @__handle_all
    def _stop_record_cameras(self, camera_names : list):
        for camera_name in camera_names:
            self.__stop_record_camera(camera_name)

    @__handle_all
    def _load_models(self, camera_model_pairs : List[Tuple[str, str]]):
        for camera_name, model_path in camera_model_pairs:
            self.__load_model(camera_name, model_path)

    @__handle_all
    def _stop_models(self, camera_names : list):
        for camera_name in camera_names:
            self.__stop_model(camera_name)

    def __perception_loop(self):
        while self.active:
            with self.change_lock:
                for camera_name, camera in self.active_cameras.items():
                    image = camera.get_latest_frame()
                    results = camera.get_latest_model_results()
                    self.latest_data[camera_name] = CameraData(image, results)
                    if camera_name in self.active_writers:
                        writer = self.active_writers[camera_name]
                        writer.write(image.frame)