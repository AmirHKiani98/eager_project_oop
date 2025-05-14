"""
Point Queue Class
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

class SpatialQueue(TrafficModel):
    """
    Class representing a point queue traffic model.
    Inherits from the TrafficModel class.
    """


    def sending_flow(self, cumulative_count_upstream, cumulative_count_downstream, dt, q_max_down):
        """
        Computes the sending flow from the point queue model.
        """
        return min(
            cumulative_count_upstream - cumulative_count_downstream,
            (q_max_down * dt).to(1).value
        )

    def receiving_flow(self, q_max_up, dt, cumulative_count_upstream, cumulative_count_downstream, link_length, k_j):
        """
        Computes the receiving flow from the point queue model.
        """
        return min(
            (k_j*link_length - (cumulative_count_upstream - cumulative_count_downstream)).to(1).value,
            (q_max_up * dt).to(1).value
            )
    

    def run(self, args):
        """
        Run the point queue model with the given arguments.
        """
        # Placeholder for running the point queue model
        q_max_up = args["q_max_up"]
        if not isinstance(q_max_up, Units.Quantity):
            raise TypeError(
                f"q_max_up should be a Units.Quantity (Per time), got {type(q_max_up)}"
            )
        
        q_max_down = args["q_max_down"]
        if not isinstance(q_max_down, Units.Quantity):
            raise TypeError(
                f"q_max_down should be a Units.Quantity (Per time), got {type(q_max_down)}"
            )
        
        next_occupancy = args["next_occupancy"]
        cumulative_count_upstream_shifted_queue = args["cumulative_count_upstream_shifted_queue"]
        cumulative_count_upstream = args["cumulative_count_upstream"]
        cumulative_count_downstream = args["cumulative_count_downstream"]
        link_length = args["link_length"]
        if not isinstance(link_length, Units.Quantity):
            raise TypeError(
                f"link_length should be a Units.Quantity (length), got {type(link_length)}"
            )

        k_j = args["k_j"]
        if not isinstance(k_j, Units.Quantity):
            raise TypeError(
                f"k_j should be a Units.Quantity (length), got {type(k_j)}"
            )

        dt = args["dt"]
        if not isinstance(dt, Units.Quantity):
            raise TypeError(
                f"dt should be a Units.Quantity (time), got {type(dt)}"
            )
        sending_flow = self.sending_flow(
            cumulative_count_upstream_shifted_queue,
            cumulative_count_downstream,
            dt,
            q_max_down
        )
        receiving_flow = self.receiving_flow(q_max_up, dt, cumulative_count_upstream, cumulative_count_downstream, link_length, k_j)
        
        
        return {
            "sending_flow": sending_flow,
            "receiving_flow": receiving_flow,
            "next_occupancy": next_occupancy
        }


