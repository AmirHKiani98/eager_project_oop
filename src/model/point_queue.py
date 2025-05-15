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

class PointQueue(TrafficModel):
    """
    Class representing a point queue traffic model.
    Inherits from the TrafficModel class.
    """
    def sending_flow(self, cummulative_count_upstream, cummulative_count_downstream, dt, q_max_down):
        """
        Computes the sending flow from the point queue model.
        """
        return min(
            cummulative_count_upstream - cummulative_count_downstream,
            (q_max_down * dt).to(1).value
        )

    def receiving_flow(self, q_max_up, dt):
        """
        Computes the receiving flow from the point queue model.
        """
        return (q_max_up * dt).to(1).value
    

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
        cummulative_count_upstream = args["cummulative_count_upstream"]
        cummulative_count_downstream = args["cummulative_count_downstream"]
        dt = args["dt"]
        if not isinstance(dt, Units.Quantity):
            raise TypeError(
                f"dt should be a Units.Quantity (time), got {type(dt)}"
            )
        sending_flow = self.sending_flow(
            cummulative_count_upstream,
            cummulative_count_downstream,
            dt,
            q_max_down
        )
        receiving_flow = self.receiving_flow(q_max_up, dt)
        
        
        return {
            "sending_flow": sending_flow,
            "receiving_flow": receiving_flow,
            "next_occupancy": next_occupancy
        }



