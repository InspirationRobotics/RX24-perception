import cv2
import time
import rclpy
from threading import Lock

from lidar_core import Lidar, LidarNode, OccupancyGrid

from comms_core import Server, Logger
from comms_core import CustomSocketMessage as csm

class TestOccupancy(Logger):

    def __init__(self):
        rclpy.init(args=None)

        self.server = Server(default_callback = self.server_callback)
        self.lidar = Lidar('combine_livox', decay_rate=0.2)
        self.lidar.add_callback(self.lidar_callback)
        self.lidar_node = LidarNode([self.lidar])

        self.gps_lock = Lock()
        self.lidar_lock = Lock()
        self.pos = [None, None, None] # lat, lon, heading
        self.lidar_data = None

        self.server.start()
        self.lidar_node.start()

    def server_callback(self, data, addr):
        data = csm.decode(data)
        lat, lon = data.get("current_position", [None, None])
        heading = data.get("current_heading", None)
        with self.gps_lock:
            self.pos = [lat, lon, heading]

    def lidar_callback(self, data):
        with self.lidar_lock:
            self.lidar_data = data

    def run(self):
        # Wait for data to be received
        while True:
            time.sleep(0.1)
            with self.gps_lock:
                if None in self.pos:
                    continue
            with self.lidar_lock:
                if self.lidar_data is None:
                    continue
            break
        og = OccupancyGrid(self.pos[0], self.pos[1], cell_size=0.2)
        frame = None
        while True:
            try:
                with self.lidar_lock:
                    lidar_data = self.lidar_data
                with self.gps_lock:
                    lat, lon, heading = self.pos
                og.update_grid(lat, lon, heading, lidar_data)
                frame = og.visualize(show=False)
                frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Occupancy Grid", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except KeyboardInterrupt:
                break
        cv2.destroyAllWindows()
        # Save the last frame
        cv2.imwrite(f"last_frame_{int(time.time())}.jpg", frame)

    def __del__(self):
        self.server.stop()
        self.lidar_node.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    to = TestOccupancy()
    to.run()