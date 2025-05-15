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
    def sending_flow(cummulative_count_upstream, cummulative_count_downstream, dt, q_max):
        """
        Computes the sending flow from the LTM model.
        """
        if not isinstance(dt, Units.Quantity):
            raise TypeError(
                f"dt should be a Units.Quantity (Time), got {type(dt)}"
            )
        if not isinstance(q_max, Units.Quantity):
            raise TypeError(
                f"q_max should be a Units.Quantity (Per time), got {type(q_max)}"
            )
        return min(
            cummulative_count_upstream - cummulative_count_downstream,
            (q_max * dt).to(1).value # type: ignore
        )
    
    @staticmethod
    def receiving_flow(q_max, dt, cummulative_count_upstream, cummulative_count_downstream, link_length, k_j):
        """
        Computes the receiving flow from the LTM model.
        """
        if not isinstance(dt, Units.Quantity):
            raise TypeError(
                f"dt should be a Units.Quantity (Time), got {type(dt)}"
            )
        if not isinstance(q_max, Units.Quantity):
            raise TypeError(
                f"q_max should be a Units.Quantity (Per time), got {type(q_max)}"
            )
        if not isinstance(link_length, Units.Quantity):
            raise TypeError(
                f"link_length should be a Units.Quantity (Length), got {type(link_length)}"
            )
        if not isinstance(k_j, Units.Quantity):
            raise TypeError(
                f"k_j should be a Units.Quantity (Density), got {type(k_j)}"
            )
        
        return min(
            (q_max * dt).to(1).value, # type: ignore
            cummulative_count_downstream + (k_j * link_length).to(1).value - cummulative_count_upstream # type: ignore
        )


    @staticmethod
    def run(args):
        """
        Run the LTM model with the given arguments.
        """
        # Required arguments
        required_args = [
            "q_max", 
            "next_occupancy", 
            "cummulative_count_downstream_receiving_flow", 
            "cummulative_count_upstream", 
            "cummulative_count_downstream", 
            "cummulative_count_downstream_receiving_flow",
            "dt",
            "link_length",
            "k_j",
            "trajectory_time",
            "link_id",
            "entry_count",
            "current_number_of_vehicles"
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
        cummulative_count_upstream_sending_flow = args["cummulative_count_downstream_receiving_flow"] # at t + dt - L/ffs
        cummulative_count_downstream = args["cummulative_count_downstream"] # at t

        cummulative_count_upstream  = args["cummulative_count_upstream"]  # at t
        cummulative_count_downstream_receiving_flow = args["cummulative_count_downstream_receiving_flow"] # at t + dt - L/wave_speed
        q_max = args["q_max"]
        dt = args["dt"]
        link_length = args["link_length"]
        k_j = args["k_j"]
        tl_status = args["tl_status"]
        entry_count = args["entry_count"]

        if not isinstance(q_max, Units.Quantity):
            raise TypeError(
                f"q_max should be a Units.Quantity (Per time), got {type(q_max)}"
            )
        if not isinstance(dt, Units.Quantity):
            raise TypeError(
                f"dt should be a Units.Quantity (Time), got {type(dt)}"
            )
        if not isinstance(link_length, Units.Quantity):
            raise TypeError(
                f"link_length should be a Units.Quantity (Length), got {type(link_length)}"
            )
        if not isinstance(k_j, Units.Quantity):
            raise TypeError(
                f"k_j should be a Units.Quantity (Density), got {type(k_j)}"
            )
        
        sending_flow = LTM.sending_flow(
            cummulative_count_upstream_sending_flow,
            cummulative_count_downstream,
            dt,
            q_max
        )
        if sending_flow < 0:
            sending_flow = 0

        if tl_status != 1:
            sending_flow = 0
        
        receiving_flow = LTM.receiving_flow(
            q_max,
            dt,
            cummulative_count_upstream,
            cummulative_count_downstream_receiving_flow,
            link_length,
            k_j
        )
        if receiving_flow < 0:
            receiving_flow = 0
        inflow = min(
            entry_count,
            receiving_flow
        )
        # Update the next occupancy
        next_occupancy = args["next_occupancy"]
        link_id = args["link_id"]
        trajectory_time = args["trajectory_time"]
        current_number_of_vehicles = args["current_number_of_vehicles"]
        return {
            "sending_flow": sending_flow,
            "receiving_flow": receiving_flow,
            "next_occupancy": next_occupancy,
            "trajectory_time": trajectory_time,
            "link_id": link_id,
            "current_number_of_vehicles": current_number_of_vehicles
        }