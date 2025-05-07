from abc import abstractmethod
from src.model.params import Parameters
from src.preprocessing.geo_loader import GeoLoader
class TrafficModel:

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
        pass
    
    