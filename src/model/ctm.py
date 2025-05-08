"""
Cell Transmission Model (CTM) for traffic flow simulation.
This module implements the Cell Transmission Model (CTM) for traffic flow simulation.
This model is a discrete-time model that simulates the flow of traffic through a series of cells.
It is designed to be used in conjunction with the TrafficModel class and provides methods for
updating cell status based on traffic density, outflows, and entry flow.
"""

import math
import numpy as np
from src.model.traffic_model import TrafficModel
from src.common_utility.units import Units
class CTM(TrafficModel):
    """
    Class representing the Cell Transmission Model (CTM) for traffic flow simulation.
    """
    # def predict(self, **kwargs):
    #     required_keys = {
    #         "time", "cell_length", "link_id", 
    #         "densities", "outflows", "entry_flow", "dt",
    #         "max_flow", "free_flow_speed", "jam_density",
    #         "wave_speed", "is_tl", "tl_status"
    #     }
    #     if not required_keys.issubset(kwargs):
    #         missing = required_keys - kwargs.keys()
    #         raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

    #     cell_length = kwargs["cell_length"]
    #     densities = kwargs["densities"]
    #     outflows = kwargs["outflows"]
    #     entry_flow = kwargs["entry_flow"]
    #     dt = kwargs["dt"]
    #     max_flow = kwargs["max_flow"]
    #     free_flow_speed = kwargs["free_flow_speed"]
    #     jam_density = kwargs["jam_density"]
    #     wave_speed = kwargs["wave_speed"]
    #     is_tl = kwargs["is_tl"]
    #     tl_status = kwargs.get("tl_status", None)

    #     num_cells = len(densities)
    #     if isinstance(densities, list):
    #         densities = np.array(densities)
    #     if isinstance(outflows, list):
    #         outflows = np.array(outflows)

    #     new_densities = densities.copy()
    #     new_outflows = outflows.copy()
    #     # dt = self.params.get_time_step(cell_length)
    #     for i in range(num_cells):
    #         if i == 0:
    #             inflow = entry_flow
    #         else:
    #             inflow = self.compute_outflow(
    #                 free_flow_speed, dt, jam_density, wave_speed,
    #                 max_flow, densities[i-1], densities[i]
    #             )

    #         if i == num_cells - 1:
    #             if is_tl and not tl_status:
    #                 outflow = 0
    #             else:
    #                 outflow = self.compute_outflow(
    #                     free_flow_speed, dt, jam_density, wave_speed,
    #                     max_flow, densities[i], None
    #                 )
    #         else:
    #             outflow = self.compute_outflow(
    #                 free_flow_speed, dt, jam_density, wave_speed,
    #                 max_flow, densities[i], densities[i+1]
    #             )

    #         new_outflows[i] = outflow
    #         new_densities[i] = densities[i] + (inflow - outflow) / cell_length

    #     return new_densities, new_outflows

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
        inflow = min(
            prev_cell_occupancy,
            flow_capacity,
            cell_capacity - current_cell_occupancy
        )
        if inflow < 0:
            raise ValueError("Inflow cannot be negative.")

        return inflow * Units.VEH


    def predict(self, **kwargs):
        """
        Predict the traffic flow using the Cell Transmission Model (CTM).

        Args:
            **kwargs: Keyword arguments containing the necessary parameters for prediction.

        Returns:
            tuple: Updated densities and outflows after applying the CTM model.
        """
        required_keys = {
            "cell_occupancy", "first_cell_inflow", "outflow", "cell_length",
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

        cell_occupancy = kwargs["cell_occupancy"]
        first_cell_inflow = kwargs["first_cell_inflow"]
        outflow = kwargs["outflow"]
        cell_length = kwargs["cell_length"]


        if not isinstance(cell_occupancy, list) and not isinstance(cell_occupancy, np.ndarray):
            raise TypeError("cell_occupancy must be a list or numpy array.")
        if isinstance(outflow, list) :
            cell_occupancy = np.array(cell_occupancy)

        if len(cell_occupancy) != len(outflow):
            raise ValueError("Length of cell_occupancy, and outflow must be the same.")
        # TODO: I user enumerate to get rid of the warning
        new_occupancy = cell_occupancy.copy()
        new_outflow = outflow.copy()
        for i, _ in enumerate(cell_occupancy):
            # First cell
            if i == 0:
                inflow = first_cell_inflow
            else:
                inflow = self.compute_flow(
                    cell_occupancy[i-1], cell_occupancy[i],
                    cell_length, self.params.flow_capacity
                )
                if i == len(cell_occupancy) - 1:
                    outflow = math.inf
                else:
                    outflow = self.compute_flow(
                        cell_occupancy[i], cell_occupancy[i+1],
                        cell_length, self.params.flow_capacity
                    )
            new_outflow[i] = outflow
            new_occupancy[i] = cell_occupancy[i] + inflow - outflow
            # In the last cell, if the outflow is greater than the occupancy, set it to 0
            if new_occupancy[i] < 0:
                new_occupancy[i] = 0
        return new_occupancy, new_outflow   
            