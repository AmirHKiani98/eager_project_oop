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
    
    @abstractmethod
    def predict(self, **args):
        """
        Abstract method to predict traffic flow.
        """
        pass
    
    