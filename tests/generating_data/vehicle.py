from math import radians, sin, cos, sqrt, atan2

class Vehicle:
    all_vehicles = {}

    def __init__(self, veh_id, car_type="Car"):
        
        # Initialize vehicle attributes
        self.lon = []
        self.lat = []
        self.time = []
        self.speed = []
        self.acc_lon = []
        self.acc_lat = []
        self.car_type = car_type
        self.track_id = len(Vehicle.all_vehicles)
        self.road_type = []  # This was likely missing before the error
        self.veh_id = veh_id
        self.link_id = []

    def add_road_type(self, road_type):
        # Adding the road type to the road_type list
        self.road_type.append(road_type)
        if road_type not in ["major", "minor", "none"]:
            raise ValueError("road_type must be one of: major, minor, none")
    

    def add_lon(self, lon):
        self.lon.append(lon)

    

    def add_lat(self, lat):
        self.lat.append(lat)

    def add_time(self, time):
        self.time.append(time)
    
    def add_speed(self, speed):
        self.speed.append(speed)
    
    def add_acc_lon(self, acc_lon):
        if acc_lon is None:
            return
        self.acc_lon.append(acc_lon)
        if len(self.acc_lon) != len(self.lon) - 1:
            raise ValueError(
                "Length of acc_lon must be equal to length of lon - 1."
                f" Current length of acc_lon: {len(self.acc_lon)}, lon: {len(self.lon)}"
                )
    
    def add_acc_lat(self, acc_lat):
        if acc_lat is None:
            return
        self.acc_lat.append(acc_lat)
        if len(self.acc_lat) != len(self.lat) - 1:
            raise ValueError(
                "Length of acc_lat must be equal to length of lat - 1."
                f" Current length of acc_lat: {len(self.acc_lat)}, lat: {len(self.lat)}"
                )

    def get_acc_lat_lon(self, acc):
        """
        Get the acceleration in the lat and lon direction.
        The acceleration is given in the direction of the vehicle.
        """
        if len(self.lon) == 0:
            raise ValueError("You are calling this method before adding any lon. That's wrong!")
        
        if len(self.lon) < 2:
            return None, None
        
        length = len(self.lon)
        vector = (self.lon[length-1] - self.lon[length-2], self.lat[length-1] - self.lat[length-2])
        vector_length = sqrt(vector[0]**2 + vector[1]**2)
        if vector_length == 0:
            return (0, 0)

        unit_vector = (vector[0] / vector_length, vector[1] / vector_length)

        acc_lon = acc * unit_vector[0]
        acc_lat = acc * unit_vector[1]
        return (acc_lon, acc_lat)

    def add_link_id(self, link_id):
        self.link_id.append(link_id)

    def get_veh_raw_data(self):
        """
        Create an object of the Vehicle class.
        Get the final row of the simulation for the vehicle.
        Its format: track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time
        track_id; type; traveled_d; avg_speed are just one number
        lat; lon; speed; lon_acc; lat_acc groups of 6
        """
        if len(self.lon) != len(self.lat):
            raise ValueError("Length of lon and lat lists must be equal")
        if len(self.lon) != len(self.time):
            raise ValueError("Length of lon and time lists must be equal")
        if len(self.lon) != len(self.speed):
            raise ValueError("Length of lon and speed lists must be equal")
        if len(self.lon) != len(self.acc_lon) - 1:
            raise ValueError("Length of lon and acc_lon lists must be equal")
        if len(self.lon) != len(self.acc_lat) - 1:
            raise ValueError("Length of lon and acc_lat lists must be equal")
        self.acc_lat.append(self.acc_lat[-1])
        self.acc_lon.append(self.acc_lon[-1])
        travel_d = self.get_tavel_distance()
        avg_speed = sum(self.speed) / len(self.speed)
        zipped = zip(self.lon, self.lat, self.time, self.speed, self.acc_lon, self.acc_lat)
        text = f"{self.track_id}; {self.car_type}; {travel_d}; {avg_speed};"
        for lon, lat, time, speed, acc, acc_lon, acc_lat in zipped:
            text += f"{lat}; {lon}; {speed}; {acc_lon}; {acc_lat}; {time};"
        return text

    def generate_processed_data(self):
        """
        Generate the processed data that follows the format after processing the raw data
        It follows the format:
        lat,lon,speed,lon_acc,lat_acc,trajectory_time,track_id,veh_type,traveled_d,avg_speed,link_id,distance_from_link,cell_id,distance_from_cell
        """
        text = ""
        avg_speed = sum(self.speed) / len(self.speed)
        travel_d = self.get_tavel_distance()
        for i in range(len(self.lon)):
            text += f"{self.lat[i]},{self.lon[i]},{self.speed[i]},{self.acc_lon[i]},{self.acc_lat[i]},{self.time[i]},{self.track_id},{self.car_type},{travel_d},{avg_speed},{self.link_id[i]}\n"
        
        return text

    def get_tavel_distance(self):
        """
        Get the traveled distance of the vehicle.
        """
        if len(self.lon) != len(self.lat):
            raise ValueError("Length of lon and lat lists must be equal")
        traveled_distance = 0
        for i in range(len(self.lon)-1):
            traveled_distance += self.get_distance(self.lat[i], self.lon[i], self.lat[i+1], self.lon[i+1])
        return traveled_distance
    
    def get_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance in meters between two coordinates using the Haversine formula.
        """

        # Radius of the Earth in meters
        R = 6371000

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Distance in meters
        distance = R * c
        return distance

    @classmethod
    def get_or_create(cls, veh_id, car_type="Car"):
        if veh_id in cls.all_vehicles:
            return cls.all_vehicles[veh_id]
        veh = cls(veh_id, car_type)
        cls.all_vehicles[veh_id] = veh
        return veh