"""
Cell Transmission Model (CTM) for traffic flow simulation.
This module implements the Cell Transmission Model (CTM) for traffic flow simulation.
This model is a discrete-time model that simulates the flow of traffic through a series of cells.
It is designed to be used in conjunction with the TrafficModel class and provides methods for
updating cell status based on traffic density, outflows, and entry flow.
"""

import math
import numpy as np
import logging
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

    def compute_flow(self, prev_cell_occupancy, current_cell_occupancy,
                       cell_length, flow_capacity):
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
        cell_capacity = self.params.get_cell_capacity(cell_length)
        if isinstance(cell_capacity, Units.Quantity):
            cell_capacity = int(cell_capacity.to(1).value)
        if isinstance(flow_capacity, Units.Quantity):
            flow_capacity = int(flow_capacity.to(1).value)
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
            "cell_occupancies", "first_cell_inflow", "cell_length",
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

        cell_occupancies = kwargs["cell_occupancies"]
        first_cell_inflow = kwargs["first_cell_inflow"]
        if not isinstance(first_cell_inflow, Units.Quantity):
            raise TypeError("first_cell_inflow must be an astropy Quantity with units")
        cell_length = kwargs["cell_length"]


        if not isinstance(cell_occupancies, list) and not isinstance(cell_occupancies, np.ndarray):
            raise TypeError("cell_occupancies must be a list or numpy array.")

        # nbbi: I user enumerate to get rid of the warning
        new_occupancy = cell_occupancies.copy()
        new_outflow = cell_occupancies.copy()
        for i, _ in enumerate(cell_occupancies):
            # First cell
            if i == 0:
                inflow = first_cell_inflow
            else:
                inflow = self.compute_flow(
                    cell_occupancies[i-1], cell_occupancies[i],
                    cell_length, self.params.flow_capacity
                )

            if i == len(cell_occupancies) - 1:
                outflow = min(self.params.get_max_flow(), cell_occupancies[i])
            else:
                outflow = self.compute_flow(
                    cell_occupancies[i], cell_occupancies[i+1],
                    cell_length, self.params.flow_capacity
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
