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

    def run(self, args):
        """
        Run the point queue model with the given arguments.
        """
        
        # Placeholder for running the point queue model
        next_occupancy = args["next_occupancy"]
        cumulative_count_upstream = args["cumulative_count_upstream"] # N(t+dt-L/vf, 0)
        cumulative_count_upstream_at_t = args["cumulative_count_upstream_at_t"] # N(t, 0)
        
        cumulative_count_downstream = args["cumulative_count_downstream"] # N(t, L)
        entry_count = args["entry_count"]
        current_number_of_vehicles = args["current_number_of_vehicles"]
        tl_status = args["tl_status"]
        trajectory_time = args["trajectory_time"]
        # if current_number_of_vehicles is None:
        #     current_number_of_vehicles = 0 # nbbi: This should not happen! Figure it out!
        link_id = args["link_id"]
        if current_number_of_vehicles is None:
            raise ValueError(
                f"current_number_of_vehicles is None for link_id {link_id}"
            )
        
        max_no_vehicles_on_link = (
            self.dl.params.q_max * self.dl.params.dt
        )
        if not isinstance(max_no_vehicles_on_link, Units.Quantity):
            raise TypeError(
                f"max_no_vehicles_on_link should be a Units.Quantity, got {type(max_no_vehicles_on_link)}"
            )
        max_no_vehicles_on_link = max_no_vehicles_on_link.to(1).value

        sending_flow = min(
            cumulative_count_upstream - cumulative_count_downstream,
            max_no_vehicles_on_link
        )
        if sending_flow < 0:
            # If the sending flow is negative, we set it to 0
            sending_flow = 0
        if not tl_status:
            # If the traffic light is red, we don't want to send any vehicles
            sending_flow = 0
        # in upstream is less than the number of vehicles in downstream
        receiving_flow = min(
            self.dl.params.jam_density_link*self.dl.geo_loader.get_link_length(link_id).to(1).value - (cumulative_count_upstream_at_t- cumulative_count_downstream),
            max_no_vehicles_on_link
            )
        if receiving_flow < 0:
            # If the receiving flow is negative, we set it to 0
            receiving_flow = 0
        new_occupancy = (
            current_number_of_vehicles 
            + min(entry_count, receiving_flow) 
            - sending_flow
        )
        if new_occupancy < 0:
            # nbbi: I added this too. Cause sometimes, the number of vehicles might be less than zero
            new_occupancy = 0
        
        return {
            "new_occupancy": new_occupancy,
            "sending_flow": sending_flow,
            "receiving_flow": receiving_flow,
            "entry_count": entry_count,
            "current_number_of_vehicles": current_number_of_vehicles,
            "cumulative_count_upstream": cumulative_count_upstream,
            "cumulative_count_downstream": cumulative_count_downstream,
            "tl_status": tl_status,
            "link_id": link_id,
            "next_occupancy": next_occupancy,
            "cumulative_count_upstream_at_t": cumulative_count_upstream_at_t,
            "trajectory_time": trajectory_time,
        }



