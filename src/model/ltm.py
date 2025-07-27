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
    def run(args):
        """
        Run the LTM model with the given arguments.
        """
        # Required arguments
        required_args = [
            "link_id",
            "cell_id",
            "x",
            "dt",
            "trajectory_time",
            "N_t_x_over_uf_0",
            "N_t_linklength_minus_x_over_wave_L",
            "N_teps_x_over_uf_0",
            "N_tpes_linklength_minus_x_over_wave_L",
            "N_t_xeps_over_uf_0",
            "N_t_linklength_minus_xeps_over_wave_L",
            "next_occupancy",
            "jam_density_link",
            "link_length",
            "eps_x",
            "eps_t",
            "next_exit"
        ]
        for arg in required_args:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
        trajectory_time = args["trajectory_time"]
        N_t_x_over_uf_0 = args["N_t_x_over_uf_0"]
        N_t_linklength_minus_x_over_wave_L = args["N_t_linklength_minus_x_over_wave_L"]
        N_teps_x_over_uf_0 = args["N_teps_x_over_uf_0"]
        N_tpes_linklength_minus_x_over_wave_L = args["N_tpes_linklength_minus_x_over_wave_L"]
        N_t_xeps_over_uf_0 = args["N_t_xeps_over_uf_0"]
        N_t_linklength_minus_xeps_over_wave_L = args["N_t_linklength_minus_xeps_over_wave_L"]
        next_occupancy = args["next_occupancy"]
        jam_density_link = args["jam_density_link"]
        link_length = args["link_length"]
        next_exit = args["next_exit"]
        cell_length = args["cell_length"]
        x = args["x"]
        dt = args['dt']
        eps_t = args["eps_t"]
        eps_x = args["eps_x"]
        
        N_t_x = min(
            N_t_x_over_uf_0,
            N_t_linklength_minus_x_over_wave_L
        )

        N_teps_x = min(
            N_teps_x_over_uf_0,
            N_tpes_linklength_minus_x_over_wave_L
        )

        N_t_xeps = min(
            N_t_xeps_over_uf_0,
            N_t_linklength_minus_xeps_over_wave_L
        )
        cell_id = args["cell_id"]
        link_id = args["link_id"]
        q = ((N_teps_x - N_t_x)/(eps_t)).to(Units.PER_HR)
        k = -1*((N_t_xeps - N_t_x)/(eps_x)).to(Units.PER_KM)

        return {
            "q": q.value,
            "k": k.value,
            "next_occupancy": next_occupancy,
            "next_k": (next_occupancy/cell_length).to(Units.PER_KM).value,
            "next_q": (next_exit/dt).to(Units.PER_HR).value,
            "link_id": link_id,
            "cell_id": cell_id,
            "x": x.to(Units.M).value
        }



        