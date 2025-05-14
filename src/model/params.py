"""
params.py
----------

This module defines the `Parameters` class, which encapsulates various traffic simulation parameters 
and provides methods to calculate key metrics such as maximum flow, time step, and jam density.

Classes:
    - Parameters: A class to manage and compute traffic simulation parameters.

Usage:
    The `Parameters` class can be instantiated with default or custom values for traffic parameters 
    such as vehicle length, free flow speed, wave speed, number of lanes, and jam density. It
    provides methods to compute the maximum flow, time step, and jam density based on the
    simulation grid's cell length.

Example:
    ```python
    params = Parameters(
        vehicle_length=5, free_flow_speed=20, wave_speed=12, num_lanes=4, jam_density_link=150
    )
    max_flow = params.max_flow(cell_length=500)
    time_step = params.get_time_step(cell_length=500)
    jam_density = params.get_jam_density(cell_length=500)
    ```
"""
import os
import json
import logging
import hashlib
from typing import Optional
from rich.logging import RichHandler
from src.common_utility.units import Units
# Configure logging
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
class Parameters():
    """
    Parameters:
        vehicle_length (float, optional): The average length of a vehicle in meters. 
            Defaults to 5.
        free_flow_speed (float, optional): The free-flow speed of vehicles in meters 
            per second. Defaults to 15.
        wave_speed (float, optional): The backward wave speed in meters per second. 
            Defaults to 10.
        num_lanes (int, optional): The number of lanes in the road. Defaults to 3.
        jam_density_link (float, optional): The jam density per link in vehicles per 
            kilometer. Defaults to 130. Doesn't include the number of lanes.
    Attributes:
        num_lanes (int): The number of lanes in the road.
        vehicle_length (float): The average length of a vehicle in meters.
        free_flow_speed (float): The free-flow speed of vehicles in meters per second.
        wave_speed (float): The backward wave speed in meters per second.
        jam_density_link (float): The jam density per link in vehicles per kilometer.
        jam_density_fd (float): The jam density for the fundamental diagram in vehicles 
            per kilometer.
        max_flow_link (float): The maximum flow per link in vehicles per second.
        critical_density (float): The critical density in vehicles per kilometer.
        tau (float): A parameter used in the calculation of wave speed. Defaults to 1.
        c (float): A constant used in the calculation of wave speed. Defaults to 10.14.
        c0 (float): A derived constant based on free flow speed and tau.
    Methods:
        max_flow(cell_length):
            Calculate the maximum flow in the system based on the fundamental diagram.
        get_time_step(cell_length):
            Calculate the time step based on the cell length and free flow speed.
        get_jam_density(cell_length):
            Calculate the jam density for a given cell length.
    """

    def __init__(
        self,
        vehicle_length  =5.0 * Units.M,
        free_flow_speed=15.0 * Units.KM_PER_HR,
        wave_speed=10.0 * Units.KM_PER_HR,
        num_lanes=3,
        jam_density_link=180.0 * Units.PER_KM,
        dt=1.0 * Units.S,
        q_max=3000.0 * Units.PER_HR,
        cache_dir=".cache"
    ):
        if not isinstance(vehicle_length, Units.Quantity):
            raise TypeError("vehicle_length must be an astropy Quantity with units")
        if not isinstance(free_flow_speed, Units.Quantity):
            raise TypeError("free_flow_speed must be an astropy Quantity with units")
        if not isinstance(wave_speed, Units.Quantity):
            raise TypeError("wave_speed must be an astropy Quantity with units")
        if not isinstance(num_lanes, int):
            raise TypeError("num_lanes must be an integer")
        if not isinstance(jam_density_link, Units.Quantity):
            raise TypeError("jam_density_link must be an astropy Quantity with units")
        if not isinstance(dt, Units.Quantity):
            raise TypeError("dt must be an astropy Quantity with units")
        if not isinstance(q_max, Units.Quantity):
            raise TypeError("q_max must be an astropy Quantity with units")
        if not isinstance(cache_dir, str):
            raise TypeError("cache_dir must be a string")
        self._is_initialized = False
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.num_lanes = num_lanes
        self.vehicle_length = vehicle_length
        self.free_flow_speed = free_flow_speed
        self.wave_speed = wave_speed
        self.dt = dt
        self.jam_density_link = jam_density_link
        self.q_max = q_max
        # Calculate the maximum number of vehicles that can flow into the system per time step.
        # nbbi: flow_capacity should be an attribute of Cell model.
        self.flow_capacity = self.q_max * self.dt # veh

        # logger.debug(
        #     "Parameters initialized with: Vehicle Length: %s, Free Flow Speed: %s, "
        #     "Wave Speed: %s, Number of Lanes: %s, Jam Density Link: %s, "
        #     "Flow Capacity: %s, Time Step: %s",
        #     self.vehicle_length, self.free_flow_speed, self.wave_speed, self.num_lanes,
        #     self.jam_density_link, self.flow_capacity, self.dt
        # )
        self._is_initialized = True
        self.save_metadata()

    def set_initialized(self, initialized: bool):
        """
        Set the initialized state of the Parameters class.

        Args:
            initialized (bool): The initialized state to set.
        """
        self._is_initialized = initialized

    def get_max_flow(self, cell_length) -> Units.Quantity:
        """
        Calculate the maximum flow in the system based on the fundamental diagram.
        
        """
        if not isinstance(cell_length, Units.Quantity):
            raise TypeError("cell_length must be an astropy Quantity with units")

        return min(
            self.q_max, # type: ignore
            min(self.free_flow_speed, self.wave_speed) # type: ignore
            * self.jam_density_link
        ) * self.dt * self.num_lanes

    def get_spatial_line_capacity(self, spatial_line_length: Units.Quantity):
        """
        Calculate the maximum number of vehicles that can be on a link based on the cell length.

        Args:
            spatial_line_length (Units.Quantity): The length of a cell in meters.

        Returns:
            float: The maximum number of vehicles that can be on the link.

        Raises:
            ValueError: If `spatial_line_length` is not provided in meters.
        """
        if not isinstance(spatial_line_length, Units.Quantity):
            raise TypeError("spatial_line_length must be an astropy Quantity with units")
        return spatial_line_length * self.jam_density_link

    def get_time_step(self, cell_length):
        """
        Calculate the time step based on the cell length and free flow speed.

        Args:
            cell_length (float): The length of a cell in the simulation grid.

        Returns:
            float: The time step, calculated as the ratio of cell length to free flow speed.
        """
        if not isinstance(cell_length, Units.Quantity):
            raise TypeError("cell_length must be an astropy Quantity with units")
        return cell_length / self.free_flow_speed

    def get_jam_density(self, cell_length: Units.Quantity):
        # Mazi, I'm not sure if this your implementation is correct. The output is
        # dimensionless. Take a look at the code below.
        """
        Calculate the jam density for a given cell length.

        Jam density is the maximum vehicle density that can be accommodated
        in a road segment without movement.

        Args:
            cell_length (float): The length of the cell (in meters).

        Returns:
            float: The jam density for the given cell length, adjusted for
            the number of lanes and scaled by the jam density per kilometer.
        """
        if not isinstance(cell_length, Units.Quantity):
            raise TypeError("cell_length must be an astropy Quantity with units")
        return self.jam_density_link * cell_length * self.num_lanes

    def get_hash_str(self, attribute_names: Optional[list] = None):
        """
        Generate a hash string based on the parameters.

        Returns:
            str: A hash string representing the parameters.
        """
        params_str = ""
        for attribute_name, attribute_value in sorted(self.__dict__.items()):
            if attribute_names is None or attribute_name in attribute_names:
                params_str += f"{attribute_name}={attribute_value}, "
        params_str = params_str.strip(", ")
        return hashlib.md5(params_str.encode()).hexdigest()

    def save_metadata(self):
        """
        Save metadata to a file.

        This method generates a hash string based on the parameters and saves it to a file.
        """
        hash_str = self.get_hash_str()
        os.makedirs(self.cache_dir + "/params", exist_ok=True)
        # if os.path.exists(
        with open(self.cache_dir + "/params/" + hash_str + ".json", "w", encoding="utf-8") as f:
            json.dump({k: str(v) for k, v in self.__dict__.items()}, f, indent=4)
        # logger.debug("Metadata saved with hash: %s", hash_str)

    def __setattr__(self, name, value):
        """
        Set an attribute of the Parameters class.

        Args:
            name (str): The name of the attribute to set.
            value: The value to set for the attribute.
        """
        # logger.debug("Setting %s to %s", name, value)

        if getattr(self, name, None) != value:
            super().__setattr__(name, value)
            if getattr(self, "_is_initialized", False) and not name.startswith("_"):
                self.save_metadata()
