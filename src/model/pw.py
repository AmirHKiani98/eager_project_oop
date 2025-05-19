"""
PW. Slightly different than what Maz wrote. 
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
            "next_occupancy", # list[unitless]
            "trajectory_time"

        ]
        for arg in required_arguments:
            if arg not in args:
                raise ValueError(f"Missing required argument: {arg}")
        densities = args["densities"]
        speeds = args["speeds"]
        
        num_cells = len(args["densities"])
        free_flow_speed = args["free_flow_speed"]
        jam_density_link = args["jam_density_link"]
        tl_status = args["tl_status"]

        epss = 0.1 * Units.PER_M # a small value
        new_densities = deepcopy(densities)
        new_speeds = deepcopy(speeds)
        next_occupancy = args["next_occupancy"]
        new_outflows = [0] * num_cells
        cell_lengths = args["cell_lengths"]

        dt = args["dt"]

        c = 10.14 # density drop
        critical_density = 150 * Units.PER_KM
        tau = 1
        c0 = free_flow_speed/(tau**2)
        inflow = 0 * Units.PER_HR
        outflow = 0 * Units.PER_HR

        for i in range(num_cells):  # iterate over all cells
            
            # find the equilibrium speed
            if 0<= densities[i] <= critical_density:
                eq_speed = free_flow_speed
            else:
                eq_speed = c * (jam_density_link/densities[i] - 1)
                eq_speed = eq_speed * Units.KM_PER_HR
            
            term = (dt.to(Units.HR).value * (eq_speed.to(Units.KM_PER_HR) - speeds[i].to(Units.KM_PER_HR))/tau).to(Units.KM_PER_HR)
            if i == 0:  # first cell
                
                new_densities[i] = densities[i] - (dt / cell_lengths[i]) * (densities[i] * speeds[i] - inflow)
                # Mazi kam aghl. Mazi kam aghl! dt * (eq_speed - speeds[i])/tau + speeds[1] both terms here should have the same units
                if len(densities) < i+2:
                    new_speeds[i] = 0
                else:
                    new_speeds[i] = speeds[i] + term - (dt / cell_lengths[i]) * c0**2 * (densities[i+1]-densities[i])/(densities[i]+ epss)

            if i == num_cells - 1:  # last cell


                # check the status of the traffic light
                if tl_status == 1: # green light
                    new_speeds[i] = speeds[i] - (dt / cell_lengths[i]) * speeds[i] * (speeds[i] - speeds[i-1]) + term
                else:    # red light
                    new_speeds[i] = 0

                new_densities[i] = densities[i] - (dt / cell_lengths[i]) * (densities[i] * speeds[i] - densities[i-1] * speeds[i-1])


            else:       # for all other cells: find density and speed using PW discrete model
                    # [i] = outflow # Maz
                new_densities[i] = densities[i] - (dt / cell_lengths[i]) * (densities[i] * speeds[i] - densities[i-1] * speeds[i-1])

                new_speeds[i] = speeds[i] - (dt / cell_lengths[i]) * speeds[i] * (speeds[i] - speeds[i-1]) + term - (dt / cell_lengths[i]) * c0**2 * (densities[i+1]-densities[i])/(densities[i]+ epss)

            new_outflows[i] = new_speeds[i] * new_densities[i] # find outflow q = kv
        density_value = []
        speed_value = []
        
        if len(new_densities) != len(cell_lengths):
            raise ValueError("Length of items:", len(new_densities), len(new_speeds), len(cell_lengths), num_cells)
        if len(new_densities) != len(next_occupancy):
            raise ValueError("Length of items:", len(new_densities), len(new_speeds), len(cell_lengths), num_cells)
        if len(new_densities) != len(new_speeds):
            raise ValueError("Length of items:", len(new_densities), len(new_speeds), len(cell_lengths), num_cells)
        for i in range(num_cells):
            density = new_densities[i]
            speed = new_speeds[i]
            if not isinstance(density, Units.Quantity):
                raise ValueError(f"Density {density} is not of type Units")
            
            if not isinstance(speed, Units.Quantity):
                speed = speed * Units.KM_PER_HR

            if not isinstance(outflow, Units.Quantity):
                raise ValueError(f"Outflow {outflow} is not of type Units")
            density = density.to(Units.PER_M).value
            speed = speed.to(Units.KM_PER_HR).value
            density_value.append(density)
            speed_value.append(speed)
        trajectory_time = args["trajectory_time"]
        return {
            "new_densities": density_value,
            "new_speeds": speed_value,
            "next_densities": next_occupancy,
            "cell_lengths": [length.to(Units.M).value for length in cell_lengths],
            "link_id": args["link_id"],
            "trajectory_time": trajectory_time,
        }
