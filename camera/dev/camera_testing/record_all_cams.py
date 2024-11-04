import cv2
import time
from camera_core import Camera, Image

camera_addrs = [
    [1,11], # right
    [1,10], # center
    [1,4], # left
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
            writer_list.append(cv2.VideoWriter(f'videos/{i}_vid_{int(time.time())}.avi', fourcc, 20.0, (1786,  953)))
        else:
            writer_list.append(None)
    return camera_list, writer_list

def run(record = True, ser = None):
    camera_list, writer_list = init_cameras(camera_addrs, record)
    for camera in camera_list:
        cv2.namedWindow(f"{camera.camera_path}", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(f"{camera.camera_path}", 640, 480)
    while camera_list[0].stream:
        last_time = time.time()
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
                ser.send("heartbeat")
                last_time = time.time()
        if cv2.waitKey(4) & 0xFF == ord('q'):
            break
    for camera, writer in zip(camera_list, writer_list):
        camera.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

def record_gps_data(record_gps : bool):
    if not record_gps:
        return None
    
    from comms_core import Server, CustomSocketMessage as csm
    def log_data(data : str, addr : str):
        data : dict = csm.decode(data)
        if data.get("current_position") is not None:
            with open("gps_data.txt", "a") as f:
                f.write(f"{time.time()}_%_{data}\n")

    ser = Server(default_callback=log_data)
    ser.start()
    return ser

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
    server = record_gps_data(record_gps)
    run(record, server)