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
    @staticmethod
    def compute_flow(kwargs):
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
            "cell_capacity", "flow_capacity",
            "alpha", "dt"
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.compute_flow(): {missing}")
        prev_cell_occupancy = kwargs["prev_cell_occupancy"]
        current_cell_occupancy = kwargs["current_cell_occupancy"]
        flow_capacity = kwargs["flow_capacity"]
        cell_capacity = kwargs["cell_capacity"]
        alpha = kwargs["alpha"]
        dt = kwargs["dt"]
        if not isinstance(dt, Units.Quantity):
            raise TypeError("dt must be a Units.Quantity")
        
        # cell_capacity = self.dl.params.get_spatial_line_capacity(cell_length)
        if isinstance(cell_capacity, Units.Quantity):
            cell_capacity = round(cell_capacity.to(1).value) # Fix: This might cause error!
        if isinstance(flow_capacity, Units.Quantity):
            flow_capacity = round(flow_capacity.to(1).value) # Fix: This might cause error!
        # alpha = self.dl.params.wave_speed/self.dl.params.wave_speed

        if isinstance(alpha, Units.Quantity):
            alpha = round(alpha.to(1).value)
        
        flow = min(
            prev_cell_occupancy,
            flow_capacity,
            alpha*(cell_capacity - current_cell_occupancy)
        ) / dt
        if flow < 0:
            raise ValueError(f"Flow cannot be negative. Check the parameters: prev_cell_occupancy= {prev_cell_occupancy},"
                             f"current_cell_occupancy= {current_cell_occupancy},"
                             f"cell_capacity= {cell_capacity},"
                             f"flow_capacity= {flow_capacity},"
                             f"alpha= {alpha}, dt= {dt}")
        return flow

    @staticmethod
    def predict(kwargs):
        """
        Predict the traffic flow using the Cell Transmission Model (CTM).

        Args:
            **kwargs: Keyword arguments containing the necessary parameters for prediction.

        Returns:
            tuple: Updated densities and outflows after applying the CTM model.
        """
        required_keys = {
            "cell_occupancies","is_tl", "tl_status",
            "cell_capacities", "flow_capacity", "q_max", "alpha",
            "inflow", "dt"
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for CTM.predict(): {missing}")

        cell_occupancies = kwargs["cell_occupancies"]
        all_inflow = kwargs["inflow"]
        cell_capacities = kwargs["cell_capacities"]
        dt = kwargs["dt"]
        flow_capacity = kwargs["flow_capacity"] # self.dl.params.flow_capacity
        q_max = kwargs["q_max"]
        alpha = kwargs["alpha"]

        if not isinstance(cell_capacities, list) and not isinstance(cell_capacities, np.ndarray):
            raise TypeError("cell_capacities must be a list or numpy array.")
        
        if not isinstance(q_max, Units.Quantity):
            raise TypeError(f"q_max must be a Units.Quantity. Type: {type(q_max)}")
        
        is_tl = kwargs["is_tl"]
        tl_status = kwargs["tl_status"]
        if not isinstance(cell_occupancies, list) and not isinstance(cell_occupancies, np.ndarray):
            raise TypeError("cell_occupancies must be a list or numpy array.")

        # nbbi: I user enumerate to get rid of the warning
        new_occupancy = cell_occupancies.copy()
        new_outflow = cell_occupancies.copy()
        

        if len(cell_occupancies) != len(cell_capacities):
            raise ValueError(f"cell_occupancies and cell_capacities must have the same length. len(cell_occupancies): {len(cell_occupancies)}, len(cell_capacities): {len(cell_capacities)}")
        for i, _ in enumerate(cell_occupancies):
            cell_capacity = cell_capacities[i]
            # First cell
            if i == 0:
                inflow = all_inflow.get("1.0", 0*Units.PER_HR)  # Default to 0 if not found
            else:
                inflow = CTM.compute_flow({
                        "prev_cell_occupancy":cell_occupancies[i-1],
                        "current_cell_occupancy":cell_occupancies[i],
                        "cell_capacity":cell_capacity,
                        "flow_capacity":flow_capacity,
                        "alpha":alpha,
                        "dt": dt
                    }
                )

            if i == len(cell_occupancies) - 1:
                if is_tl and tl_status:
                    outflow = 0*Units.PER_HR  # No outflow if traffic light is red
                else:
                    # max_flow = self.dl.params.get_max_flow(cell_len)
                    if not isinstance(q_max, Units.Quantity):
                        raise TypeError("max_flow must be a Units.Quantity")
                    max_flow_value = q_max.to(Units.PER_HR).value
                    cell_flow_value = (cell_occupancies[i] / dt).to(Units.PER_HR).value
                    outflow = min(max_flow_value, cell_flow_value)*Units.PER_HR
            else:
                outflow = CTM.compute_flow({
                        "prev_cell_occupancy":cell_occupancies[i],
                        "current_cell_occupancy":cell_occupancies[i+1],
                        "cell_capacity":cell_capacity,
                        "flow_capacity":flow_capacity,
                        "alpha":alpha,
                        "dt": dt
                    }
                )
            if not isinstance(inflow, Units.Quantity):
                raise TypeError("inflow must be a Units.Quantity", f"Type: {type(inflow)}", f"Value: {inflow}")
            if not isinstance(outflow, Units.Quantity):
                # outflow = int(outflow.to(Units.PER_HR).value)
                raise TypeError("outflow must be a Units.Quantity", f"Type: {type(outflow)}", f"Value: {outflow}")
            new_outflow[i] = outflow.to(Units.PER_HR).value
            
            new_occupancy[i] = cell_occupancies[i] + ((inflow - outflow) * dt).to(1).value
            # In the last cell, if the outflow is greater than the occupancy, set it to 0
            if new_occupancy[i] < 0:
                new_occupancy[i] = 0
        return new_occupancy, new_outflow

    @staticmethod
    def run(args):
        """
        Run the traffic model with the provided arguments.

        Args:
            args (list): A list of arguments containing the necessary parameters 
                 for running the model.
        Returns:
            None
        """
        occupancy_list = args["occupancy_list"]
        inflow = args["inflow"]
        is_tl = args["is_tl"]
        tl_status = args["tl_status"]
        trajectory_time = args["trajectory_time"]
        next_occupancy = args["next_occupancy"]
        link_id = args["link_id"]
        cell_capacities = args["cell_capacities"]
        flow_capacity = args["flow_capacity"]
        q_max = args["q_max"]
        alpha = args["alpha"]
        dt = args["dt"]
        if len(occupancy_list) != len(cell_capacities):
            print(f"DEBUG: cell_occupancies: {occupancy_list}, len(cell_occupancies): {len(occupancy_list)}")
            print(f"DEBUG: cell_capacities: {cell_capacities}, len(cell_capacities): {len(cell_capacities)}")
            print(f"DEBUG: all_inflow: {inflow}, len(all_inflow): {len(inflow)}")
            print(f"DEBUG: Running CTM for link_id: {link_id}, trajectory_time: {trajectory_time}")
        if not isinstance(occupancy_list, list) and not isinstance(occupancy_list, np.ndarray):
            raise TypeError("occupancy_list must be a list or numpy array.")

        if not isinstance(is_tl, bool):
            raise TypeError("is_tl must be a boolean")
        if tl_status == 1:
            tl_status = True
        elif tl_status == 0:
            tl_status = False
        if not isinstance(tl_status, bool):
            raise TypeError("tl_status must be a boolean")
        
        # Debug information
        # print(f"DEBUG: Running CTM for link_id: {link_id}, trajectory_time: {trajectory_time}")
        # print(f"DEBUG: occupancy_list length: {len(occupancy_list)}")
        # print(f"DEBUG: cell_capacities length: {len(cell_capacities)}")
        # print(f"DEBUG: occupancy_list: {occupancy_list}")
        # print(f"DEBUG: cell_capacities: {cell_capacities}")
        
        new_occupancy, new_outflow = CTM.predict({
            "cell_occupancies": occupancy_list,
            "inflow": inflow,
            "is_tl": is_tl,
            "tl_status": tl_status,
            "cell_capacities": cell_capacities,
            "flow_capacity": flow_capacity,
            "q_max": q_max,
            "alpha": alpha,
            "dt": dt
        })

        return {
            "occupancy_list": occupancy_list,
            "inflow": {cell_id: inflow.to(Units.PER_HR).value for cell_id, inflow in inflow.items()},
            "link_id": link_id,
            "is_tl": is_tl,
            "tl_status": tl_status,
            "new_occupancy": new_occupancy,
            "new_outflow": new_outflow,
            "trajectory_time": trajectory_time,
            "next_occupancy": next_occupancy
        }
