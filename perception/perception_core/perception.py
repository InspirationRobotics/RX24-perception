from camera_core import Camera
import numpy as np
import cv2

class Perception:
    def __init__(self, camera_addresses: List[Tuple[int, int]]):
        """
        Initialize the Perception Stack
        :param camera_list: List containing the Camera Objects
        :param camera_addresses: List containing tuples of camera addresses
        :param model_list: List containing the relevant models
        :param is_streaming: Boolean that indicates whether we are streaming 
        """
        self.camera_list = []
        self.camera_addresses = camera_addresses
        self.model_list = []
        self.is_streaming = False

    def init_cameras(self):
        """
        Initialize the Cameras
        """
        for addr in self.camera_addresses:
            camera = Camera(bus_addr=addr, camera_type='port')
            camera.warmup()
            camera.start()
            self.camera_list.append(camera)
        print("Cameras initialized.")

    def load_model(self, model_path: str, camera_index: int = None):
        """
        Load the model onto all cameras or a specific camera.

        :param model_path: Path to the model.
        :param camera_index: Index of the camera to load the model onto. If None, loads onto all cameras.
        """
        if camera_index is not None:
            if 0 <= camera_index < len(self.camera_list):
                # Load model onto a specific camera
                model = ML_Model(model_path, "tensorrt")
                self.camera_list[camera_index].load_model_object(model)
                self.model_list.append((camera_index, model))
                print(f"Model loaded onto camera {camera_index}")
            else:
                print(f"Error: Camera index {camera_index} is out of range.")
        else:
            # Load model onto all cameras
            for i, camera in enumerate(self.camera_list):
                model = ML_Model(model_path, "tensorrt")
                camera.load_model_object(model)
                self.model_list.append((i, model))
            print("Model loaded onto all cameras.")

    def get_latest_frames(self):
        """
        Get the latest frames from all cameras.

        :return: A list of the latest frames from each camera.
        """
        frames = []
        for camera in self.camera_list:
            frame = camera.get_latest_frame(undistort=True, with_cuda=True)
            if frame:
                frames.append(frame.frame)
        return frames

    def start_stream(self):
        """
        Start streaming from all cameras.
        """
        self.is_streaming = True
        for camera in self.camera_list:
            camera.start_stream()

    def stop_stream(self):
        """
        Stop streaming from all cameras.
        """
        self.is_streaming = False
        for camera in self.camera_list:
            camera.stop_stream()

    def draw_model_results(self):
        """
        Draw model results on the frames.
        """
        frames_with_results = []
        for camera in self.camera_list:
            frame = camera.get_latest_frame()
            if frame is not None:
                frame = camera.draw_model_results(frame.frame)
                frames_with_results.append(frame)
        return frames_with_results

    def stitch_frames(self, frames: list, x_distance: float, y_angle: float, fov_overlap: float) -> np.ndarray:
        """
        Stitches the frames of the three cameras together into one panoramic frame
        NOTE: this is implemented currently to attempt to take advantage of the fixed position of the cameras.
        NOTE: it currently does not use the x_distance value. If using the fixed distance does not work, we can also attempt to stitch by features
        :param frames: list of the individual frames
        :param x_distance: distance between the cameras in meters
        :param y_angle: angle between the cameras in degrees
        :param fov_overlap: percentage value of the overlap between two of the cameras
        """
        # TODO: this will take a lot of tweaking and maybe a full re-implemntation
        if len(frames) != 3:
            raise ValueError("Expected exactly 3 frames for stitching.")

        # Calculate transformation based on the angle Y
        angle_rad = np.radians(y_angle)

        # Create homography for left camera (rotating by -Y degrees)
        h_left = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad), 0],
                           [np.sin(-angle_rad),  np.cos(-angle_rad), 0],
                           [0, 0, 1]])

        # Create homography for right camera (rotating by +Y degrees)
        h_right = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                            [np.sin(angle_rad),  np.cos(angle_rad), 0],
                            [0, 0, 1]])

        # Warp the left and right images using the calculated homographies
        height, width = frames[1].shape[:2]
        warped_left = cv2.warpPerspective(frames[0], h_left, (width, height))
        warped_right = cv2.warpPerspective(frames[2], h_right, (width, height))

        # Create an empty canvas for the final panoramic image
        canvas_width = width * 3 - int(width * fov_overlap) * 2  # Adjust for overlap
        panorama = np.zeros((height, canvas_width, 3), dtype=np.uint8)

        # Overlay the images without blending
        # Place the left frame on the left part of the canvas
        panorama[:, :width] = warped_left

        # Overlay the center frame in the middle
        panorama[:, width - int(width * fov_overlap):width*2 - int(width * fov_overlap)] = frames[1]

        # Overlay the right frame on the right part of the canvas
        panorama[:, canvas_width-width:] = warped_right

        return panorama
