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
from abc import abstractmethod
from src.model.params import Parameters
from src.preprocessing.geo_loader import GeoLoader
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
    def __init__(self, geo_loader: GeoLoader, params: Parameters):
        self.geo_loader = geo_loader
        self.params = params

    def set_params(self, params):
        """
        Set the parameters for the traffic model.
        
        Args:
            params (Parameters): Parameters object containing traffic model parameters.
        """
        self.params = params

    def get_cell_length(self, cell_id, link_id):
        """
        Get the length of a specific cell.

        Args:
            cell_id (int): The ID of the cell.
            link_id (int): The ID of the link.

        Returns:
            float: Length of the specified cell.
        """
        return self.geo_loader.get_cell_length(cell_id, link_id)

    @abstractmethod
    def predict(self, **args):
        """
        Abstract method to predict traffic flow.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    