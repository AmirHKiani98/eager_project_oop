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
    def count_ltm(cumulative_upstream, cumulative_downstream, x, wave_speed, dt, link_length, jam_density):
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

        if not isinstance(jam_density, Units.Quantity):
            raise TypeError("jam_density must be a Quantity")
        target1 = cumulative_upstream
        congestion_intersect_x1 = x + wave_speed * dt
        if congestion_intersect_x1>= 0 and congestion_intersect_x1<= link_length:
            congestion_intersect_x = congestion_intersect_x1

        else:
            congestion_intersect_x = link_length
        
        target2 = jam_density * (congestion_intersect_x - x) + cumulative_downstream
        return min(target1, target2)


        

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
            "x",
            "wave_speed",
            "jam_density_link"
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
        x = args["x"]
        if not isinstance(x, Units.Quantity):
            raise TypeError("x must be a Quantity")
        
        eps_x = 0.01 * Units.M
        eps_t = 0.01 * Units.S
        # LTM_count(time, segment_id, entry_flow, location, horizon)
        n1 = LTM.count_ltm(
            cumulative_upstream=args["upstream_value_freeflow"],
            cumulative_downstream=args["downstream_value_freeflow"],
            x=args["x"],
            wave_speed=args["wave_speed"],
            dt=args["dt"],
            link_length=args["link_length"],
            jam_density=args["jam_density_link"]
        )
        # LTM_count(time, segment_id, entry_flow, location+eps, horizon)
        n2 = LTM.count_ltm(
            cumulative_upstream=args["upstream_value_freeflow_with_eps_x"],
            cumulative_downstream=args["downstream_value_freeflow_with_eps_x"],
            x=args["x"] + eps_x,
            wave_speed=args["wave_speed"],
            dt=args["dt"],
            link_length=args["link_length"],
            jam_density=args["jam_density_link"]
        )

        # LTM_count(time, segment_id, entry_flow, location, horizon+eps)
        n3 = LTM.count_ltm(
            cumulative_upstream=args["upstream_value_freeflow_with_eps_t"],
            cumulative_downstream=args["downstream_value_freeflow_with_eps_t"],
            x=args["x"],
            wave_speed=args["wave_speed"],
            dt=args["dt"] + eps_t,
            link_length=args["link_length"],
            jam_density=args["jam_density_link"]
        )
        density = (n1 - n2)/eps_x
        flow = -(n1 - n3)/eps_t
        return {
            "density": density,
            "flow": flow,
            "link_id": args["link_id"],
            "trajectory_time": args["trajectory_time"],
            "cell_id": args["cell_id"],
            "link_length": args["link_length"].to(Units.M),
            "x": args["x"].to(Units.M),
            "next_occupancy": args["next_occupancy"],
        }