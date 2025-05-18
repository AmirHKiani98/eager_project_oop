"""
Implementation of the LTM traffic model
This code has be inspired by Maziar Zamanpour original code in colab:
https://colab.research.google.com/drive/1GCNJirX8MgCMaBJ03yjMXGz26yfK_RwI#scrollTo=FzfLO1MM4iLr
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
    def count_ltm(cumulative_upstream, cumulative_downstream, x, wave_speed, dt, link_length):
        """
        Count the number of vehicles in the LTM model.
        """
        if not isinstance(x, Units.Quantity):
            raise TypeError("x must be a Quantity")
        if not isinstance(wave_speed, Units.Quantity):
            raise TypeError("wave_speed must be a Quantity")

        if not isinstance(dt, Units.Quantity):
            raise TypeError("dt must be a Quantity")
        if not isinstance(link_length, Units.Quantity):
            raise TypeError("link_length must be a Quantity")
        
        congestion_intersect_x1 = x + wave_speed * dt
        if congestion_intersect_x1>= 0 and congestion_intersect_x1<= link_length:
            congestion_intersect_x = congestion_intersect_x1
        

    @staticmethod
    def run(args):
        """
        Run the LTM model with the given arguments.
        """
        # Required arguments
        required_args = [
            "link_id",
            "trajectory_time",
            "upstream_value_freeflow_with_eps_x",
            "downstream_value_freeflow_with_eps_x",
            "upstream_value_freeflow_with_eps_t",
            "downstream_value_freeflow_with_eps_t",
            "upstream_value_freeflow",
            "downstream_value_freeflow",
            "upstream_value_wavespeed_with_eps_x",
            "downstream_value_wavespeed_with_eps_x",
            "upstream_value_wavespeed_with_eps_t",
            "downstream_value_wavespeed_with_eps_t",
            "upstream_value_wavespeed",
            "downstream_value_wavespeed",
            "cell_id",
            "link_length",
            "x"
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
        # LTM_count(time, segment_id, entry_flow, location, horizon)
        
        
        return {
            
        }