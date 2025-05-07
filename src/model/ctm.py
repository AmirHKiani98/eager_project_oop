"""
Cell Transmission Model (CTM) for traffic flow simulation.
This module implements the Cell Transmission Model (CTM) for traffic flow simulation.
This model is a discrete-time model that simulates the flow of traffic through a series of cells.
It is designed to be used in conjunction with the TrafficModel class and provides methods for
updating cell status based on traffic density, outflows, and entry flow.
"""

import math
from src.model.traffic_model import TrafficModel
class CTM(TrafficModel):
    """
    Class representing the Cell Transmission Model (CTM) for traffic flow simulation.
    """


    def predict(self, **kwargs):
        required_keys = {"time", "cell_id", "link_id", "densities", "outflows", "entry_flow"}
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

        time = kwargs["time"]
        cell_id = kwargs["cell_id"]
        link_id = kwargs["link_id"]
        densities = kwargs["densities"]
        outflows = kwargs["outflows"]
        entry_flow = kwargs["entry_flow"]
        num_cells = len(densities)
        new_densities = densities.copy()
        new_outflows = outflows.copy()
        cell_length = self.get_cell_length(cell_id, link_id)
        dt = self.params.get_time_step(cell_length)
        for i in range(num_cells):
            if i == 0: 
                inflow = entry_flow
            else:
                inflow = min(
                    self.params.max_flow(cell_length),
                    self.params.free_flow_speed * densities[i-1] * dt,
                    self.params.wave_speed * (
                        self.params.get_jam_density(cell_length) - densities[i]
                    ) * dt
                )

            if i == num_cells - 1:  # last cell
                # check if there is a traffic light at the end of the segment
                if self.is_tl(link_id):
                    # check the status of the traffic light
                    if self.tl_status(time, link_id): # green light
                        outflow = min(
                            self.params.max_flow(cell_length),
                            self.params.free_flow_speed * densities[i] * dt,
                            math.inf
                        )
                        new_outflows[i] = outflow
                    else:
                        outflow = 0
                        new_outflows[i] = outflow
                else:
                    outflow = min(
                        self.params.max_flow(cell_length),
                        self.params.free_flow_speed * densities[i] * dt,
                        math.inf
                    )
                    new_outflows[i] = outflow # Maz
            else:  # for all other cells: minimum of max flow and the flow to the next cell
                outflow = min(
                    self.params.max_flow(cell_length),
                    self.params.free_flow_speed * densities[i] * dt,
                    self.params.wave_speed * (
                        self.params.get_jam_density(cell_length) - densities[i+1]
                    ) * dt
                )
                new_outflows[i] = outflow # Maz
            new_densities[i] = densities[i] + (
                (inflow - outflow) / cell_length
            )  # n(t+1) = n(t) + (y(i) - y(i+1))/dx

        return new_densities, new_outflows # Maz
