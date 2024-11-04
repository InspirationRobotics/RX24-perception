import cv2
import time
from pathlib import Path
from camera_core import Camera, Image

from comms_core import Server, CustomSocketMessage as csm


camera_addrs = [
    [1,11], # right
    [1,10], # center
    [1,4], # left
]

def init_cameras(addrs : list, record : bool = True, timestamp : int = int(time.time())):
    camera_list : list[Camera] = []
    writer_list : list[cv2.VideoWriter] = []
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    for addr in addrs:
        camera_list.append(Camera(bus_addr=addr, camera_type='port'))
    else:
        folder_path = None
    for i, camera in enumerate(camera_list):
        camera.camera_name = str(i)
        camera.warmup()
        camera.start()
        if record:
            folder_path = Path(f"recordings_{timestamp}")
            folder_path.mkdir(exist_ok=True)
            writer_list.append(cv2.VideoWriter(f'{folder_path}/{i}_vid_{timestamp}.avi', fourcc, 20.0, (1786,  953)))
        else:
            writer_list.append(None)
    return camera_list, writer_list

def run(record = True, ser : Server = None, timestamp = int(time.time())):
    camera_list, writer_list = init_cameras(camera_addrs, record, timestamp)
    for camera in camera_list:
        cv2.namedWindow(f"{camera.camera_path}", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(f"{camera.camera_path}", 640, 480)
    last_time = time.time()
    while camera_list[0].stream:
        for camera, writer in zip(camera_list, writer_list):
            frame : Image = camera.get_latest_frame()
            if frame is None: continue
            frame = frame.frame
            # Draw timestamp on frame
            curr_time = time.time()
            cv2.putText(frame, f"{curr_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(f"{camera.camera_path}", frame)
            if writer is not None:
                writer.write(frame)
            if time.time() - last_time > 0.2 and ser is not None:
                # heartbeat
                data = {"heartbeat" : time.time()}
                ser.send(csm.encode(data))
                last_time = time.time()
        if cv2.waitKey(4) & 0xFF == ord('q'):
            break
    for camera, writer in zip(camera_list, writer_list):
        camera.stop()
        if writer is not None:
            writer.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.stop()

def record_gps_data(record_gps : bool):
    if not record_gps:
        return None
    
    timestamp = int(time.time())
    folder_path = Path(f"recordings_{timestamp}")
    folder_path.mkdir(exist_ok=True)
    def log_data(data : str, addr : str):
        data : dict = csm.decode(data)
        if data.get("current_position") is not None:
            with open(f"{folder_path}/gps_data_{timestamp}.txt", "a") as f:
                msg_pos = data.get("current_position")
                msg_head = data.get("current_heading")
                f.write(f"{time.time()}_%_{msg_pos}_%_{msg_head}\n")

    ser = Server(default_callback=log_data)
    ser.start()
    return ser, timestamp

if __name__ == "__main__":
    record = input("Record? (y/n)")
    if record in ["y", "yes"]:
        print("Recording all cameras...")
        record = True
        rec_gps = input("Record GPS data? (y/n)")
        if rec_gps in ["y", "yes"]:
            print("Recording GPS data...")
            record_gps = True
        else:
            print("Not recording GPS data...")
            record_gps = False
    else:
        print("Not recording...")
        record = False
        record_gps = False
    server, timestamp = record_gps_data(record_gps)
    run(record, server, timestamp)