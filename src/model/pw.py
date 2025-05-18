"""
PW 
"""

from copy import deepcopy
from src.model.traffic_model import TrafficModel
from src.common_utility.units import Units

class PW(TrafficModel):

    @staticmethod
    def run(args):

        """
        Run the PW model.
        """
        # Required arguments
        required_arguments = [
            "densities", # list[Units]
            "speeds", # list[Units]
            "cell_lengths", # list[Units]
            "dt", # Units
            "free_flow_speed", # Units
            "jam_density_link", # Units
            "tl_status", 

        ]
        densities = args["densities"]
        speeds = args["speeds"]
        num_cells = len(args["densities"])
        free_flow_speed = args["free_flow_speed"]
        jam_density_link = args["jam_density_link"]
        tl_status = args["tl_status"]

        epss = 0.1 # a small value
        new_densities = deepcopy(densities)
        new_speeds = deepcopy(speeds)
        new_outflows = [0] * num_cells
        cell_length = args["cell_lengths"]
        dt = args["dt"]

        c = 10.14 # density drop
        critical_density = 150 * Units.PER_KM
        tau = 1
        c0 = free_flow_speed/(tau**2)
        inflow = 0
        outflow = 0

        for i in range(num_cells):  # iterate over all cells
        
            # find the equilibrium speed
            if 0<= densities[i] <= critical_density:
                eq_speed = free_flow_speed
            else:
                eq_speed = c * (jam_density_link/densities[i] - 1)

            if i == 0:  # first cell
                new_densities[i] = densities[i] - (dt / cell_length) * (densities[i] * speeds[i] - inflow)

                new_speeds[i] = speeds[i] + dt * (eq_speed - speeds[i])/tau - (dt / cell_length) * c0**2 * (densities[i+1]-densities[i])/(densities[i]+ epss)

            if i == num_cells - 1:  # last cell


                # check the status of the traffic light
                if tl_status == 1: # green light
                    new_speeds[i] = speeds[i] - (dt / cell_length) * speeds[i] * (speeds[i] - speeds[i-1]) + dt * (eq_speed - speeds[i])/tau
                else:    # red light
                    new_speeds[i] = 0

                else:
                new_speeds[i] = speeds[i] - (dt / cell_length) * speeds[i] * (speeds[i] - speeds[i-1]) + dt * (eq_speed - speeds[i])/tau

                new_densities[i] = densities[i] - (dt / cell_length) * (densities[i] * speeds[i] - densities[i-1] * speeds[i-1])


            else:       # for all other cells: find density and speed using PW discrete model
                    # [i] = outflow # Maz
                new_densities[i] = densities[i] - (dt / cell_length) * (densities[i] * speeds[i] - densities[i-1] * speeds[i-1])

                new_speeds[i] = speeds[i] - (dt / cell_length) * speeds[i] * (speeds[i] - speeds[i-1]) + dt * (eq_speed - speeds[i])/tau - (dt / cell_length) * c0**2 * (densities[i+1]-densities[i])/(densities[i]+ epss)

            new_outflows[i] = new_speeds[i] * new_densities[i] # find outflow q = kv

        for i in range(num_cells):
            density = new_densities[i]
            speed = new_speeds[i]
            outflow = new_outflows[i]
            if not isinstance(density, Units):
                raise ValueError(f"Density {density} is not of type Units")
        
            if not isinstance(speed, Units):
                raise ValueError(f"Speed {speed} is not of type Units")

            if not isinstance(outflow, Units):
                raise ValueError(f"Outflow {outflow} is not of type Units")

        return new_densities, new_outflows , new_speeds
