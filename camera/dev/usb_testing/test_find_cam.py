from camera_core import FindCamera
import time

fc = FindCamera()
cam = fc.find_cam(1,6)

print(cam)
