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
from src.common_utility.units import Units
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
            kilometer. Defaults to 130.
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
        self, vehicle_length=5.0, free_flow_speed=15.0, wave_speed=10.0,
        num_lanes=3, jam_density_link=180.0, dt=1.0, q_max=3000.0
    ):
        self.num_lanes = num_lanes
        self.vehicle_length = vehicle_length * Units.M # m
        self.free_flow_speed = free_flow_speed * Units.KM_PER_HR # km/h
        self.wave_speed = wave_speed * Units.KM_PER_HR # km/h
        self.dt = dt * Units.S # seconds
        self.jam_density_link = jam_density_link * Units.VEH_PER_KM # veh/km # Should be around 180
        self.q_max = q_max * Units.VEH_PER_HR
        # Calculate the maximum number of vehicles that can flow into the system per time step.
        # TODO: flow_capacity should be an attribute of Cell model.
        self.flow_capacity = self.q_max / self.dt


    def get_max_flow(self, cell_length):
        """
        Calculate the maximum flow in the system based on the fundamental diagram.

        Returns:
            float: Maximum flow (vehicles/second).
        """
        max_flow = min(
            self.q_max,
            min(self.free_flow_speed, self.wave_speed)
            * self.get_jam_density(cell_length)
            * self.num_lanes
        )
        return max_flow

    def get_cell_capacity(self, cell_length: Units.Quantity):
        """
        Calculate the maximum number of vehicles that can be on a link based on the cell length.

        Args:
            cell_length (Units.Quantity): The length of a cell in meters.

        Returns:
            float: The maximum number of vehicles that can be on the link.

        Raises:
            ValueError: If `cell_length` is not provided in meters.
        """
        if not isinstance(cell_length, Units.Quantity):
            raise TypeError("cell_length must be an astropy Quantity with units")
        return self.jam_density_link / cell_length

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
        return self.jam_density_link/1000 * cell_length * self.num_lanes
