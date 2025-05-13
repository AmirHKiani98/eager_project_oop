"""
A module for gererating test data for unit tests.
"""
from math import atan2, degrees
from pathlib import Path
from shapely.geometry import Point as POINT
import numpy as np
from src.common_utility.units import Units
class Generator():
    """
    A class to generate test data for unit tests.
    """

    def __init__(
        self, 
        number_raw_data: int = 1000, 
        p1: POINT = POINT(0, 0), 
        x: float = 0.09527999999997677, 
        no_links: int = 5
    ):
        """
        Initializes the Generator class.
        """
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.number_raw_data = number_raw_data
        self.p1 = p1
        self.p2 = POINT(x, -1*x)
        self.no_links = no_links
        self.generating_intersection_data()
        self.minor_roads = self.generate_minor_road()
        self.all_track_ids = []
        self.timeinterval = 0.04 * Units.S
        self.green_time = 60 * Units.S
        self.sim_time = 1000 * Units.S

        

    def generating_raw_data(self):
        """
        Generates raw data for testing purposes.
        the track_id with even number is a vehicle on minor road
        the track_id with odd number is a vehicle on major road
        """
        # Between p1 and p2 there is 1500 meters distance
        lon_diff = self.p2.x - self.p1.x
        
        
    
    def generate_lat_lon(self,
                         is_minor_road: bool = False):
        """
        Generates random latitude and longitude values.
        """
        if is_minor_road:
            lon =  np.linspace(self.p1.x, self.p2.x, num=np.random.randint(20, 50))
            lat = np.linspace(self.p1.y, self.p2.y, num=np.random.randint(20, 50))
            return lon, lat, is_minor_road
        else:
            random_minor_road = np.random.choice(self.minor_roads)
            lon = np.linspace(random_minor_road[0].x, random_minor_road[1].x, num=np.random.randint(20, 50))
            lat = np.linspace(random_minor_road[0].y, random_minor_road[1].y, num=np.random.randint(20, 50))
            return lon, lat, is_minor_road
    
    def generate_speeds(self, lat: list | np.ndarray, lon: list | np.ndarray):
        """
        Generates random speed values.
        """
        # Generate random speed values between 0 and 100 km/h
        if len(lat) != len(lon):
            raise ValueError("Length of lat and lon must be the same")
        
        speeds = []

        for index in range(len(lat)):
            pt1 = POINT(lon[index], lat[index])
            pt2 = POINT(lon[index+1], lat[index+1])
            angle, distance = self.get_degree_and_distance(pt1, pt2)
            # Calculate speed based on distance and time
            speed = distance / self.timeinterval
            speeds.append(speed)
        
        # last speed
        speeds.append(np.mean(speeds))

        
        

    
    def generate_type(self):
        """
        Generates random vehicle types.
        """
        return np.random.choice(["Car", "Motorcycle", "Medium Vehicle", "Taxi", "Heavy Vehicle"])

    def calculate_traveled_d(self, pt1:POINT, pt2:POINT):
        """
        Calculates the distance between two points.
        """
        return self.get_degree_and_distance(pt1, pt2)[1]

    def calculate_average_speed(self, speed:list):
        """
        Calculates the average speed.
        """
        return np.mean(speed)
    

    def generating_intersection_data(self):
        """
        Generates traffic signal data for testing purposes.
        """
        
        # Between p1 and p2 there is 1500 meters distance
        lon_diff = self.p2.x - self.p1.x  # x is longitude
        lat_diff = self.p2.y - self.p1.y  # y is latitude
        
        # Generate 5 points between p1 and p2
        points = [POINT(self.p1.x + lon_diff * (i / self.no_links), self.p1.y + lat_diff * (i / self.no_links)) for i in range(1, 6)]
        np.random.seed(10)

        self.intersections = points
        # Get 5 minor roads on this line
        
        
        text = "lon,lat"
        for index in range(len(points)):
            text += f"\n{points[index].x},{points[index].y}"

        with open(self.base_dir / "tests/assets/traffic_signal_data.csv", "w") as f:
            f.write(text)
    

    def run(self):
        """
        Generates random simulation time.
        """
        pass
        

    
    def get_degree_and_distance(self, p1: POINT, p2:POINT):
        """
        Get the angle (bearing) in degrees and the distance between two points in meters.
        """
        # Calculate the distance in degrees
        distance = p1.distance(p2)
        
        # Convert to meters (assuming WGS84, 1 degree ~ 111.32 km)
        distance_meters = distance * 111320  # meters per degree

        # Calculate the angle (bearing)
        delta_lon = p2.x - p1.x
        delta_lat = p2.y - p1.y
        angle = degrees(atan2(delta_lat, delta_lon))
        return angle, distance_meters

    def random_point_on_line(self, p1: POINT, p2: POINT):
        t = np.random.uniform(0, 1)
        x = p1.x + t * (p2.x - p1.x)
        y = p1.y + t * (p2.y - p1.y)
        return POINT(x, y)

    def generate_minor_road(self, no_minor_road: int = 3):
        """
        Generates minor road points.
        """
        lines = []
        for index in range(no_minor_road):
            # Generate a random point on the line between p1 and p2
            random_point = self.random_point_on_line(self.p1, self.p2)
            # Append the random point to the list of minor roads
            intercept = self.get_intercept(random_point)
            x1 = random_point.x * 1.01
            y1 = x1 + intercept
            x2 = random_point.x * 0.99
            y2 = x2 + intercept
            # Create a new point with the calculated coordinates
            pt1 = POINT(x1, y1)
            pt2 = POINT(x2, y2)
            lines.append((pt1, pt2))

        return lines

    def generate_traffic_light_status(self):
        """
        Generates random traffic light status.
        """
        # Generate random traffic light status
        result = []
        first_end_time = np.random.randint(10, 20)
        first_type = np.random.choice([True, False])
        for time in np.arange(0, first_end_time, step=self.timeinterval):
            result.append({"trajectory_time": time, "traffic_light_status":first_type})
        return result
        
    
    def get_intercept(self, pt: POINT):
        """
        Calculate the intercept point based on the given point.
        """
        # Calculate the intercept point based on the given point
        intercept = pt.y - pt.y
        return intercept
            

    

    
        
    
if __name__ == "__main__":
    generator = Generator()
    # generator.generating_raw_data()
    generator.generating_intersection_data()