import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

'''
A class which builds an occupancy grid as the robot moves through the environment.
It relies on an origin lat/lon which is centered at (0,0).

To update the grid, it takes in the current lat/lon, current heading, and the lidar data.

The lidar data is reduced to only points where z < 0, z > -1.5, x < 30, and y < 30, y > -30.
This makes it essentially a 30m x 30m x 1.5m box positioned directly in front of the robot (since the lidars are mounted on the front of the robot).
The 1.5m height is to remove points on the ground and above the robot. The 30m x 30m box is to remove points that are too far away from the robot.
The 1.5m is then also condensed into one plane by overlaying the points on the x-y plane.

The grid is a dictionary where the key is the x,y coordinate and the value is the occupancy value.
As the robot moves, the grid can expand in any direction.

The grid is initialized with -1 in each cell to represent unknown occupancy.
The higher the value of a cell (max 100), the more certain we are that the cell is occupied.
When the robot moves, the local grid is overlayed onto the global grid at the robot's current position and heading.
'''

class Grid:
    '''
    The issue with using an array as a grid is that even if the grid is sparse, it still takes up a lot of memory. 
    (Ex. if we wanted just an L shape, it would still be a full square).
    To solve this, we can use a dictionary where the key is the x,y coordinate and the value is the occupancy value.
    We can use dunder methods to make it act like a 2D array.
    '''
    def __init__(self, max_value=100, min_value=0, default_value=-1):
        self.grid = {}
        self.max_value = max_value
        self.min_value = min_value
        self.default_value = default_value

        self.x_range = [0,0] # [min, max]
        self.y_range = [0,0] # [min, max]

    def increment(self, x_list, y_list, value):
        if len(x_list) != len(y_list):
            raise ValueError('Mismatched x y indices.')
        for i in range(len(x_list)):
            self[(x_list[i], y_list[i])] += value

    def visualize(self, opencv = True, show = True):
        '''
        Visualizes the grid by returning a frame like representation using Numpy.
        The whiter the cell, the higher the occupancy value.
        '''
        if not opencv and show:
            x_coords = []
            y_coords = []
            for (x, y) in self.grid.keys():
                if self.grid[(x, y)] < 1:
                    continue
                x_coords.append(x)
                y_coords.append(y)
            plt.figure(figsize=(8, 8))  # Adjust figure size if needed
            plt.scatter(x_coords, y_coords, color='black', marker='o')
            plt.grid(True)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        if opencv:
            x_size = self.x_range[1] - self.x_range[0] + 1 + 60
            y_size = self.y_range[1] - self.y_range[0] + 1 + 60
            frame = np.zeros((y_size, x_size))
            for coord in self.grid:
                x = coord[0] - self.x_range[0] + 30
                y = y_size - (coord[1] - self.y_range[0] + 30)
                frame[y, x] = self.grid[coord] / self.max_value

            frame = np.array(frame * 255, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if show:
                show_frame = cv2.resize(frame, (800,800), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Occupancy Grid", show_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return frame

    def _handle_key(self, key, func, *args):
        result = []
        if type(key) != tuple or len(key) != 2:
            raise ValueError('Key must be a tuple of length 2.')
        if type(key[0]) == list and type(key[1]) == list:
            if len(key[0]) != len(key[1]):
                raise ValueError('Mismatched x y indices.')
            for i in range(len(key[0])):
                res = func((key[0][i], key[1][i]), *args)
                if res != None:
                    result.append(res)
            return result
        result = func(key, *args)
        return result

    def _adjust_ranges(self, key):
        self.x_range[0] = min(self.x_range[0], key[0])
        self.x_range[1] = max(self.x_range[1], key[0])
        self.y_range[0] = min(self.y_range[0], key[1])
        self.y_range[1] = max(self.y_range[1], key[1])

    def __getitem__(self, key):
        return self._handle_key(key, self.grid.get, self.default_value)

    def __setitem__(self, key, value):
        self._adjust_ranges(key)
        value = max(min(value, self.max_value), self.min_value)
        self._handle_key(key, self.grid.__setitem__, value)

    def __delitem__(self, key):
        self._handle_key(key, self.grid.__delitem__)

    def __len__(self):
        return len(self.grid)

    def __contains__(self, key):
        return key in self.grid
    
    def __add__(self, other):
        new_grid = Grid()
        for key in self.grid:
            new_grid[key] = self.grid[key]
        for key in other.grid:
            new_grid[key] += other.grid[key]
        return new_grid
    
    def __iadd__(self, other):
        for key in other.grid:
            self[key] += other.grid[key]
        return self

    def __eq__(self, other):
        return self.grid == other.grid
    
    def __iter__(self):
        return iter(self.grid)
    
    def __str__(self):
        return str(self.grid)
    
    def __repr__(self):
        return repr(self.grid)


class OccupancyGrid:

    def __init__(self, origin_lat, origin_lon, cell_size=0.5):
        self.origin = (origin_lat, origin_lon)
        self.cell_size = cell_size
        self.grid = Grid()

    def _global_to_local(self, lat, lon):
        '''
        This assumes that the origin is at the center of the grid. (0,0)
        This also assumes that positive x is North, positive y is East.
        '''
        x = (lat - self.origin[0]) * 111139
        y = (lon - self.origin[1]) * 111139 * math.cos(math.radians(lat))
        x = int(x / self.cell_size)
        y = int(y / self.cell_size)
        return x, y
    
    def _local_to_global(self, x, y):
        lat = x * self.cell_size / 111139 + self.origin[0]
        lon = y * self.cell_size / (111139 * math.cos(math.radians(lat))) + self.origin[1]
        return lat, lon
    
    def _clean_lidar_data(self, lidar_data):
        '''
        Removes points that are too far away from the robot or are above the robot.
        '''
        clean_data = []
        for point_cloud in lidar_data:
            clean_data.append(point_cloud[(point_cloud[:,2] < 0) & (point_cloud[:,2] > -1.5) # Z axis
                                          & (point_cloud[:,1] < 15) & (point_cloud[:,1] > -15) # Y axis
                                          & (point_cloud[:,0] < 30)]) # X axis
        return clean_data

    def _process_local_grid(self, lidar_data):
        '''
        Processes the lidar data into a smaller 30mx30m local grid. 
        The robot is horizontally centered in the local grid, and placed at the bottom of the grid.
        This grid will later be added onto the global grid at the robot's current position and heading.
        '''
        local_grid = Grid(min_value=-10)
        clean_data = self._clean_lidar_data(lidar_data)
        # Overlay the points onto the local grid.
        # For the lidar data, +x is forward, +y is left, so we need to swap them and invert y.
        for point_cloud in clean_data:
            x_coords = -point_cloud[:,1]
            y_coords = point_cloud[:,0]
            x_indices = np.floor(x_coords / self.cell_size).astype(int)
            y_indices = np.floor(y_coords / self.cell_size).astype(int)
            local_grid.increment(x_indices, y_indices, 5)

        # Decrement all the values that were not detected (make more efficient?)
        # x can go from 0m -> 30m, y can go from -15m -> 15m
        local_grid_size = int(30 // self.cell_size)
        for x in range(0, local_grid_size):
            for y in range(-local_grid_size // 2, local_grid_size // 2):
                if (x, y) not in local_grid:
                    local_grid[(x, y)] = -10

        return local_grid
    
    def _orient_local_grid(self, local_grid : Grid, heading, offset):
        '''
        Orients the local grid to match the robot's heading and position.
        The robot's heading is 0 when it is facing North.
        The local grid is oriented with x as right and y as forward.
        '''
        theta = math.radians(heading)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        oriented_grid = Grid()
        for coord in local_grid:
            x, y = np.dot(rotation_matrix, np.array(list(coord)))
            x += offset[0]
            y += offset[1]
            oriented_grid[int(x), int(y)] = local_grid[coord]
        return oriented_grid

    def update_grid(self, lat, lon, heading, lidar_data):
        '''
        Updates the global grid with the current lidar data.
        '''
        offset = self._global_to_local(lat, lon)
        local_grid = self._process_local_grid(lidar_data)
        # Rotate + Translate the local grid to match the robot's heading + Position.
        oriented_grid = self._orient_local_grid(local_grid, heading, offset)
        # Add the oriented grid to the global grid.
        self.grid += oriented_grid

    def get_grid(self):
        return self.grid
    
    def get_global_grid(self):
        global_grid = Grid()
        for coord in self.grid:
            lat, lon = self._local_to_global(coord[0], coord[1])
            global_grid[(lat, lon)] = self.grid[coord]
        return global_grid

    def visualize(self):
        self.grid.visualize()

'''
Just for testing
'''
def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000 # Radius of earth in meters. Use 3956 for miles
    return c * r

def bearing(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate the bearing between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # calculate the bearing
    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)
    solved_bearing = math.degrees(math.atan2(y, x))
    solved_bearing = (solved_bearing + 360) % 360
    return solved_bearing
'''
================
'''

if __name__ == '__main__':
    og = OccupancyGrid(37.7749, -122.4194)
    print(og._global_to_local(37.7749, -122.4194))
    lat1, lon1 = og._local_to_global(50, 50)
    lat2, lon2 = og._local_to_global(50, 51)
    print(haversine(lat1, lon1, lat2, lon2))
    print(bearing(lat1, lon1, lat2, lon2))
    pass