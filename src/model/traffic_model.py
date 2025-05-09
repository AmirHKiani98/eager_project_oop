"""
traffic_model.py
This module defines the `TrafficModel` abstract base class, which serves as a 
blueprint for implementing traffic simulation models. It provides a structured 
interface for setting parameters, retrieving cell lengths, and predicting 
traffic flow.

Classes:
    TrafficModel: An abstract base class for traffic simulation models.

Usage:
    This module is intended to be extended by specific implementations of 
    traffic models. Subclasses must implement the `predict` method to define 
    the logic for traffic flow prediction.
Dependencies:
    - abc.abstractmethod: Used to define abstract methods.
    - src.model.params.Parameters: Represents configuration and settings for 
      the traffic model.
    - src.preprocessing.geo_loader.GeoLoader: Provides geographical data 
      related to the traffic model.
"""
import math
from abc import abstractmethod
from src.preprocessing.data_loader import DataLoader
class TrafficModel:
    """
    TrafficModel is an abstract base class that represents a traffic simulation model. 
    It provides a structure for defining traffic-related operations, such as setting parameters, 
    retrieving cell lengths, and predicting traffic flow. This class is designed to be extended 
    by specific implementations of traffic models.
    Attributes:
        geo_loader (GeoLoader): An instance of GeoLoader used to retrieve geographical data 
            related to the traffic model, such as cell lengths.
        params (Parameters): An instance of Parameters containing configuration and settings 
            for the traffic model.
    Methods:
        set_params(params):
            Updates the parameters of the traffic model with a new Parameters object.
        get_cell_length(cell_id, link_id):
            Retrieves the length of a specific cell within a link using the GeoLoader instance.
        predict(**args):
            Abstract method that must be implemented by subclasses to predict traffic flow 
            based on the model's logic and input arguments.
    """
    def __init__(self, dl: DataLoader):
        """
        Initialize the TrafficModel with a GeoLoader instance, Parameters object,
        and a DataLoader instance.
        Args:
            geo_loader (GeoLoader): An instance of GeoLoader for geographical data.
            params (Parameters): An instance of Parameters for model configuration.
            dl (DataLoader): An instance of DataLoader for loading data.
        """
        self.dl = dl

    def get_cell_length(self, cell_id, link_id):
        """
        Get the length of a specific cell.

        Args:
            cell_id (int): The ID of the cell.
            link_id (int): The ID of the link.

        Returns:
            float: Length of the specified cell.
        """
        return self.dl.geo_loader.get_cell_length(cell_id, link_id)

    def is_tl(self, link_id):
        """
        Check if a link has a traffic light.

        Args:
            link_id (int): The ID of the link.

        Returns:
            bool: True if the link has a traffic light, False otherwise.
        """
        return self.dl.is_tl(link_id)

    def tl_status(self, time, link_id):
        """
        Get the status of a traffic light.

        Args:
            time (int): The current time.
            link_id (int): The ID of the link.

        Returns:
            int: Status of the traffic light (1 for green, 0 for red).
        """
        return self.dl.tl_status(time, link_id)

    def compute_outflow(
        self,
        free_flow_speed, dt, jam_density, wave_speed,
        max_flow, density_current, density_next=None
    ):
        """
        Computes the outflow of traffic based on the given parameters.

        Parameters:
            free_flow_speed (float): The speed of traffic under free-flow 
            conditions (e.g., vehicles per unit time).
            dt (float): The time step duration.
            jam_density (float): The maximum vehicle density (e.g., vehicles 
            per unit length) at which traffic is completely jammed.
            wave_speed (float): The speed at which traffic congestion 
            propagates backward.
            density_current (float): The current traffic density.
            density_next (float, optional): The traffic density at the next 
            location. Defaults to None.

        Returns:
            float: The computed outflow, which is the minimum of the maximum 
            flow, demand, and supply.
        """
        demand = free_flow_speed * density_current * dt
        if density_next is None:
            supply = math.inf
        else:
            supply = wave_speed * (jam_density - density_next) * dt
        return min(max_flow, demand, supply)

    @abstractmethod
    def predict(self, **args):
        """
        Abstract method to predict traffic flow.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def run(self, trajectory_timestamp, **args):
        """
        Abstract method to run the traffic model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def compute_flow(self, **args):
        """
        Abstract method to compute flow.
        """
        raise NotImplementedError("Subclasses must implement this method.")
