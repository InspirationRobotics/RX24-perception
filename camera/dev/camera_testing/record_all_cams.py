import cv2
from camera_core import Camera, Image

camera_addrs = [
    [1,11], # port
    [1,4], # center
    [1,10], # right
]

def init_cameras(addrs : list, record : bool = True):
    camera_list : list[Camera] = []
    writer_list : list[cv2.VideoWriter] = []
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    for addr in addrs:
        camera_list.append(Camera(bus_addr=addr, camera_type='port'))
    for i, camera in enumerate(camera_list):
        camera.camera_name = str(i)
        camera.warmup()
        camera.start()
        if record:
            writer_list.append(cv2.VideoWriter(f'videos/2_{i}.avi', fourcc, 20.0, (1786,  953)))
        else:
            writer_list.append(None)
    return camera_list, writer_list

def run(record = True):
    camera_list, writer_list = init_cameras(camera_addrs, record)
    for camera in camera_list:
        cv2.namedWindow(f"{camera.camera_path}", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(f"{camera.camera_path}", 640, 480)
    while camera_list[0].stream:
        for camera, writer in zip(camera_list, writer_list):
            frame : Image = camera.get_latest_frame()
            if frame is None: continue
            frame = frame.frame
            cv2.imshow(f"{camera.camera_path}", frame)
            if writer is not None:
                writer.write(frame)
        if cv2.waitKey(4) & 0xFF == ord('q'):
            break
    for camera, writer in zip(camera_list, writer_list):
        camera.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run(False)