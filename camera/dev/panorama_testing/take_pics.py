import cv2
import time
from pathlib import Path
from camera_core import Camera, Image


CURRENT_FILE_PATH = Path(__file__).parent.absolute()
IMG_FILE_PATH = CURRENT_FILE_PATH / "sample_images"

if not IMG_FILE_PATH.exists():
    IMG_FILE_PATH.mkdir()

port = Camera(bus_addr=[1,8], camera_type='port') #port
starboard = Camera(bus_addr=[1,7], camera_type='starboard') #starboard

port.warmup()
starboard.warmup()

port.start()
starboard.start()

cv2.namedWindow("port", cv2.WINDOW_NORMAL)
cv2.resizeWindow("port", 640, 480)

cv2.namedWindow("starboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("starboard", 640, 480)

i = 0
while port.stream and starboard.stream:
    port_frame : Image = port.get_latest_frame(undistort=True, with_cuda=True)
    starboard_frame : Image = starboard.get_latest_frame(undistort=True, with_cuda=True)
    if port_frame is None or starboard_frame is None:
        continue
    port_frame = port_frame.frame
    starboard_frame = starboard_frame.frame

    cv2.imshow("port", port_frame)
    cv2.imshow("starboard", starboard_frame)
    k = cv2.waitKey(4) & 0xFF

    if k == ord('q'):
        print("Exiting...")
        break

    elif k == ord('s'):
        i += 1
        port_name = f"port_{i}.jpg"
        starboard_name = f"starboard_{i}.jpg"
        print(f'saving image')
        cv2.imwrite(str(IMG_FILE_PATH / port_name), port_frame)
        cv2.imwrite(str(IMG_FILE_PATH / starboard_name), starboard_frame)

    if i == 5:
        print("Max number of images saved")
        break

port.stop()
starboard.stop()

cv2.destroyAllWindows()