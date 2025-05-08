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
class CTM(TrafficModel):
    """
    Class representing the Cell Transmission Model (CTM) for traffic flow simulation.
    """
    def predict(self, **kwargs):
        required_keys = {
            "time", "cell_length", "link_id", 
            "densities", "outflows", "entry_flow", "dt",
            "max_flow", "free_flow_speed", "jam_density",
            "wave_speed", "is_tl", "tl_status"
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

        cell_length = kwargs["cell_length"]
        densities = kwargs["densities"]
        outflows = kwargs["outflows"]
        entry_flow = kwargs["entry_flow"]
        dt = kwargs["dt"]
        max_flow = kwargs["max_flow"]
        free_flow_speed = kwargs["free_flow_speed"]
        jam_density = kwargs["jam_density"]
        wave_speed = kwargs["wave_speed"]
        is_tl = kwargs["is_tl"]
        tl_status = kwargs.get("tl_status", None)

        num_cells = len(densities)
        if isinstance(densities, list):
            densities = np.array(densities)
        if isinstance(outflows, list):
            outflows = np.array(outflows)

        new_densities = densities.copy()
        new_outflows = outflows.copy()
        # dt = self.params.get_time_step(cell_length)
        for i in range(num_cells):
            if i == 0:
                inflow = entry_flow
            else:
                inflow = self.compute_outflow(
                    free_flow_speed, dt, jam_density, wave_speed,
                    max_flow, densities[i-1], densities[i]
                )

            if i == num_cells - 1:
                if is_tl and not tl_status:
                    outflow = 0
                else:
                    outflow = self.compute_outflow(
                        free_flow_speed, dt, jam_density, wave_speed,
                        max_flow, densities[i], None
                    )
            else:
                outflow = self.compute_outflow(
                    free_flow_speed, dt, jam_density, wave_speed,
                    max_flow, densities[i], densities[i+1]
                )

            new_outflows[i] = outflow
            new_densities[i] = densities[i] + (inflow - outflow) / cell_length

        return new_densities, new_outflows
