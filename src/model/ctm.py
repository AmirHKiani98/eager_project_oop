"""
Cell Transmission Model (CTM) for traffic flow simulation.
This module implements the Cell Transmission Model (CTM) for traffic flow simulation.
This model is a discrete-time model that simulates the flow of traffic through a series of cells.
It is designed to be used in conjunction with the TrafficModel class and provides methods for
updating cell status based on traffic density, outflows, and entry flow.
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

class CTM(TrafficModel):
    """
    Class representing the Cell Transmission Model (CTM) for traffic flow simulation.
    """

    def compute_flow(self, **kwargs):
        """
        Computes the inflow into a cell based on the previous cell's occupancy 
        and the current cell's occupancy. The inflow is limited by the maximum 
        number of vehicles that can be in the cell and the maximum number of 
        vehicles that can flow into the cell.
        cell_capacity: Maximum number of vehicles that can be in the cell = N_i
        flow_capacity: Maximum number of vehicles that can flow into the cell = Q_i
        returns:
            inflow: The number of vehicles that can flow into the cell. Type: Units.Quantity
        """
        required_keys = {
            "prev_cell_occupancy", "current_cell_occupancy",
                       "cell_length", "flow_capacity"
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.compute_flow(): {missing}")
        prev_cell_occupancy = kwargs["prev_cell_occupancy"]
        current_cell_occupancy = kwargs["current_cell_occupancy"]
        cell_length = kwargs["cell_length"]
        flow_capacity = kwargs["flow_capacity"]

        cell_capacity = self.dl.params.get_cell_capacity(cell_length)
        if isinstance(cell_capacity, Units.Quantity):
            cell_capacity = round(cell_capacity.to(1).value) # Fix: This might cause error!
        if isinstance(flow_capacity, Units.Quantity):
            flow_capacity = round(flow_capacity.to(1).value) # Fix: This might cause error!

        inflow = min(
            prev_cell_occupancy,
            flow_capacity,
            cell_capacity - current_cell_occupancy
        )
        if inflow < 0:
            raise ValueError("Inflow cannot be negative.")

        return inflow


    def predict(self, **kwargs):
        """
        Predict the traffic flow using the Cell Transmission Model (CTM).

        Args:
            **kwargs: Keyword arguments containing the necessary parameters for prediction.

        Returns:
            tuple: Updated densities and outflows after applying the CTM model.
        """
        required_keys = {
            "cell_occupancies", "first_cell_inflow", "link_id", "is_tl", "tl_status"
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

        cell_occupancies = kwargs["cell_occupancies"]
        first_cell_inflow = kwargs["first_cell_inflow"]
        if not isinstance(first_cell_inflow, Units.Quantity):
            raise TypeError("first_cell_inflow must be an astropy Quantity with units")

        link_id = kwargs["link_id"]
        is_tl = kwargs["is_tl"]
        tl_status = kwargs["tl_status"]
        if not isinstance(cell_occupancies, list) and not isinstance(cell_occupancies, np.ndarray):
            raise TypeError("cell_occupancies must be a list or numpy array.")

        # nbbi: I user enumerate to get rid of the warning
        new_occupancy = cell_occupancies.copy()
        new_outflow = cell_occupancies.copy()
        for i, _ in enumerate(cell_occupancies):
            cell_length = self.dl.geo_loader.get_cell_length(
                cell_id=i+1,
                link_id=link_id
            )
            # First cell
            if i == 0:
                inflow = first_cell_inflow
            else:
                inflow = self.compute_flow(
                    prev_cell_occupancy=cell_occupancies[i-1],
                    current_cell_occupancy=cell_occupancies[i],
                    cell_length=cell_length,
                    flow_capacity=self.dl.params.flow_capacity
                )

            if i == len(cell_occupancies) - 1:
                if is_tl and tl_status:
                    outflow = 0
                else:
                    max_flow = self.dl.params.get_max_flow().to(1).value
                    outflow = min(max_flow, cell_occupancies[i])
            else:
                outflow = self.compute_flow(
                    prev_cell_occupancy=cell_occupancies[i],
                    current_cell_occupancy=cell_occupancies[i+1],
                    cell_length=cell_length,
                    flow_capacity=self.dl.params.flow_capacity
                )
            if isinstance(inflow, Units.Quantity):
                inflow = int(inflow.to(1).value)
            if isinstance(outflow, Units.Quantity):
                outflow = int(outflow.to(1).value)

            new_outflow[i] = outflow
            new_occupancy[i] = cell_occupancies[i] + inflow - outflow

            # In the last cell, if the outflow is greater than the occupancy, set it to 0
            if new_occupancy[i] < 0:
                new_occupancy[i] = 0

        return new_occupancy, new_outflow

    def run(self, args):
        """
        Run the traffic model with the provided arguments.

        Args:
            args (list): A list of arguments containing the necessary parameters 
                 for running the model.
        Returns:
            None
        """
        occupancy_list, first_cell_inflow, link_id, is_tl, tl_status = args
        if not isinstance(occupancy_list, list) and not isinstance(occupancy_list, np.ndarray):
            raise TypeError("occupancy_list must be a list or numpy array.")
        if not isinstance(first_cell_inflow, Units.Quantity):
            raise TypeError("first_cell_inflow must be an astropy Quantity with units")
        if not isinstance(link_id, int):
            raise TypeError("link_id must be an integer")
        if not isinstance(is_tl, bool):
            raise TypeError("is_tl must be a boolean")
        if not isinstance(tl_status, bool):
            raise TypeError("tl_status must be a boolean")
        new_occupancy, new_outflow = self.predict(
            cell_occupancies=occupancy_list,
            first_cell_inflow=first_cell_inflow,
            link_id=link_id,
            is_tl=is_tl,
            tl_status=tl_status
        )
        return {
            "occupancy_list": occupancy_list,
            "first_cell_inflow": first_cell_inflow,
            "link_id": link_id,
            "is_tl": is_tl,
            "tl_status": tl_status,
            "new_occupancy": new_occupancy,
            "new_outflow": new_outflow
        }
