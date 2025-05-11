"""
Implementation of the LTM traffic model
"""
import logging
import numpy as np
from rich.logging import RichHandler
from src.model.traffic_model import TrafficModel
from src.common_utility.units import Units
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

class LTM(TrafficModel):
    """
    Class representing a LTM traffic model.
    """

    def predict(self, **args):
        """
        Predicts the outcome of the LTM
        """
        pass

    
    def run(self, args):
        """
        Run the LTM model with the given arguments.
        """
        n_tlxw = args["n_tlxw"]
        n_txvf = args["n_txvf"]
        
