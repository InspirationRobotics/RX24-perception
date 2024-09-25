import math
import numpy as np

'''
A class which builds an occupancy grid as the robot moves through the environment.
It relies on an origin lat/lon which is centered at grid_size/2, grid_size/2.

To update the grid, it takes in the current lat/lon, current heading, and the lidar data.

The lidar data is reduced to only points where z < 0, z > -1.5, x < 30, and y < 30, y > -30.
This makes it essentially a 30m x 30m x 1.5m box positioned directly in front of the robot (since the lidars are mounted on the front of the robot).
The 1.5m height is to remove points on the ground and above the robot. The 30m x 30m box is to remove points that are too far away from the robot.
The 1.5m is then also condensed into one plane by overlaying the points on the x-y plane.

The grid is a 2D numpy array where each cell represents a 0.5m x 0.5m square in the environment.
The grid starts with a size of 100 x 100 cells, centered at the origin. (50m x 50m)
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
    def __init__(self, max_value=100, default_value=-1):
        self.grid = {}
        self.max_value = max_value
        self.default_value = default_value

    def __getitem__(self, key):
        return self.grid.get(key, self.default_value)

    def __setitem__(self, key, value):
        if type(key) != tuple or len(key) != 2:
            raise ValueError('Key must be a tuple of length 2.')
        value = min(value, self.max_value)
        self.grid.__setitem__(key, value)

    def __delitem__(self, key):
        self.grid.__delitem__(key)

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
            clean_data.append(point_cloud[(point_cloud[:,2] < 0) & (point_cloud[:,2] > -1.5) & (point_cloud[:,0] < 30) & (point_cloud[:,1] < 30) & (point_cloud[:,1] > -30)])
        return clean_data

    def _process_local_grid(self, lidar_data):
        '''
        Processes the lidar data into a smaller 30mx30m local grid. 
        The robot is horizontally centered in the local grid, and placed at the bottom of the grid.
        This grid will later be added onto the global grid at the robot's current position and heading.
        '''
        local_grid_size = int(30 / self.cell_size)
        local_grid = np.full((local_grid_size, local_grid_size), -1)
        center = (local_grid_size // 2, 0) # The robot is horizontally centered in the local grid (assumes 0,0 is bottom left)
        clean_data = self._clean_lidar_data(lidar_data)
        # Overlay the points onto the local grid.
        # For the lidar data, x is forward, y is right. So we flip y and x.
        for point_cloud in clean_data:
            x_coords = point_cloud[:,1]
            y_coords = point_cloud[:,0]
            x_indices = np.floor(x_coords / self.cell_size + center[0]).astype(int)
            y_indices = np.floor(y_coords / self.cell_size + center[1]).astype(int)
            mask = (x_indices >= 0) & (x_indices < local_grid_size) & (y_indices >= 0) & (y_indices < local_grid_size)
            x_indices = x_indices[mask]
            y_indices = y_indices[mask]
            local_grid[x_indices, y_indices] = 1

        return local_grid

    def update_grid(self, lat, lon, heading, lidar_data):
        pass



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
    # og = OccupancyGrid(37.7749, -122.4194)
    # print(og._global_to_local(37.7749, -122.4194))
    # lat1, lon1 = og._local_to_global(50, 50)
    # lat2, lon2 = og._local_to_global(50, 51)
    # print(haversine(lat1, lon1, lat2, lon2))
    # print(bearing(lat1, lon1, lat2, lon2))
    grid = Grid()
    grid[0,0] = 10
    grid[0,0] += 200
    print(grid[0,0])
    pass