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
    @staticmethod
    def sending_flow(cummulative_count_upstream_offset, cummulative_count_downstream, dt, q_max_down):
        """
        Computes the sending flow from the point queue model.
        """
        return min(
            cummulative_count_upstream_offset - cummulative_count_downstream,
            (q_max_down * dt).to(1).value
        )

    @staticmethod
    def receiving_flow(q_max_up, dt):
        """
        Computes the receiving flow from the point queue model.
        """
        return (q_max_up * dt).to(1).value
    
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
            "cummulative_count_downstream",
            "dt",
            "tl_status",
            "entry_count",
            "trajectory_time",
            "link_id",
            "current_number_of_vehicles"
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
        cummulative_count_downstream = args["cummulative_count_downstream"]
        dt = args["dt"]

        if not isinstance(dt, Units.Quantity):
            raise TypeError(
                f"dt should be a Units.Quantity (time), got {type(dt)}"
            )
        sending_flow = PointQueue.sending_flow(
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
        receiving_flow = PointQueue.receiving_flow(q_max_up, dt)

        link_id = args["link_id"]
        trajectory_time = args["trajectory_time"]
        entry_count = args["entry_count"]
        # todo not sure if this is correct
        inflow = min(
            entry_count,
            receiving_flow
        )
        current_number_of_vehicles = args["current_number_of_vehicles"]
        return {
            "sending_flow": sending_flow,
            "receiving_flow": receiving_flow,
            "next_occupancy": next_occupancy,
            "trajectory_time": trajectory_time,
            "link_id": link_id,
            "inflow": inflow,
            "current_number_of_vehicles": current_number_of_vehicles
        }



