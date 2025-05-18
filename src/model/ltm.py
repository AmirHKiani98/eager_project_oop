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

    @staticmethod
    def run(args):
        """
        Run the LTM model with the given arguments.
        """
        # Required arguments
        required_args = [
            "link_id",
            "trajectory_time",
            "dt",
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
        
        
        return {
            
        }