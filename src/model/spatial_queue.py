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

    @staticmethod
    def sending_flow(cummulative_count_upstream, cummulative_count_downstream, dt, q_max_down):
        """
        Computes the sending flow from the point queue model.
        """
        return min(
            cummulative_count_upstream - cummulative_count_downstream,
            (q_max_down * dt).to(1).value
        )

    @staticmethod
    def receiving_flow(q_max_up, dt, cummulative_count_upstream, cummulative_count_downstream, link_length, k_j):
        """
        Computes the receiving flow from the point queue model.
        """
        return min(
            (k_j*link_length - (cummulative_count_upstream - cummulative_count_downstream)).to(1).value,
            (q_max_up * dt).to(1).value
            )
    
    @staticmethod
    def run(args):
        """
        Run the point queue model with the given arguments.
        """
        # Required arguments
        required_args = [
            "q_max_up", 
            "q_max_down", 
            "next_occupancy", 
            "cummulative_count_upstream_offset", 
            "cummulative_count_upstream", 
            "cummulative_count_downstream", 
            "dt", 
            "trajectory_time",
            "link_id",
            "tl_status",
            "entry_count",
            "current_number_of_vehicles",
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
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
        cummulative_count_upstream_offset = args["cummulative_count_upstream_offset"]
        cummulative_count_upstream = args["cummulative_count_upstream"]
        cummulative_count_downstream = args["cummulative_count_downstream"]
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
        sending_flow = SpatialQueue.sending_flow(
            cummulative_count_upstream_offset,
            cummulative_count_downstream,
            dt,
            q_max_down
        )
        
        if sending_flow < 0:
            sending_flow = 0
        tl_status = args["tl_status"]
        if tl_status != 1:
            sending_flow = 0
        receiving_flow = SpatialQueue.receiving_flow(q_max_up, dt, cummulative_count_upstream, cummulative_count_downstream, link_length, k_j)
        trajectory_time = args["trajectory_time"]
        link_id = args["link_id"]
        entry_count = args["entry_count"]
        # todo not sure if this is correct
        inflow = min(
            entry_count,
            receiving_flow
        )
        current_number_of_vehicles = args["current_number_of_vehicles"]
        new_occupancy = next_occupancy + inflow - sending_flow
        return {
            "outflow": sending_flow, # already applied filteration on sending flow so it became outflow
            "receiving_flow": receiving_flow,
            "next_occupancy": next_occupancy,
            "trajectory_time": trajectory_time,
            "link_id": link_id,
            "inflow": inflow,
            "current_number_of_vehicles": current_number_of_vehicles,
            "new_occupancy": new_occupancy,
        }


