import cv2
import time
import rclpy
from threading import Thread, Lock

from lidar_core import Lidar, LidarNode, OccupancyGrid

from comms_core import Server, Logger
from comms_core import CustomSocketMessage as csm

class TestOccupancy(Logger):

    def __init__(self):
        super().__init__('TestOccupancy')
        rclpy.init(args=None)

        self.server = Server(default_callback = self.server_callback)
        self.lidar = Lidar('combined_lidar', decay_rate=0.2)
        self.lidar_node = LidarNode([self.lidar])

        self.gps_lock = Lock()
        self.lidar_lock = Lock()
        self.pos = [None, None, None] # lat, lon, heading
        self.lidar_data = None

        self.server.start()
        self.lidar_node.start()
        self.active = True
        self.heartbeat_thread = Thread(target=self.send_hearbeat_thread, daemon=True)
        self.heartbeat_thread.start()

    def server_callback(self, data, addr):

        data = csm.decode(data)
        pos = data.get("current_position", None)
        if pos is not None:
            lat, lon = pos
            heading = data.get("current_heading", None)
            if heading is not None:
                with self.gps_lock:
                    self.pos = [lat, lon, heading]
        self.log("GPS data received")

    def run(self):
        # Wait for data to be received
        while True:
            self.log("Waiting for data...")
            time.sleep(0.1)
            with self.gps_lock:
                if None in self.pos:
                    continue
            break
        self.log("Data received. Starting visualization...")
        with self.gps_lock:
            og = OccupancyGrid(self.pos[0], self.pos[1], cell_size=0.2)
        frame = None
        while True:
            try:
                lidar_data = self.lidar.get_points_np()
                with self.gps_lock:
                    lat, lon, heading = self.pos
                start_time = time.time()
                og.update_grid(lat, lon, heading, lidar_data)
                print("Time taken for update:", time.time()- start_time)
                frame = og.visualize(show=False)
                frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Occupancy Grid", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print("Time taken per frame:", time.time()- start_time)
            except KeyboardInterrupt:
                break
        cv2.destroyAllWindows()
        # Save the last frame
        cv2.imwrite(f"last_frame_{int(time.time())}.jpg", frame)

    def send_hearbeat_thread(self):
      msg = {}
      while self.active:
            msg['heartbeat'] = 00
            self.server.send(csm.encode(msg))
            time.sleep(1.5)

    def __del__(self):
        self.active = False
        self.heartbeat_thread.join(3)
        self.server.stop()
        self.lidar_node.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    to = TestOccupancy()
    to.run()