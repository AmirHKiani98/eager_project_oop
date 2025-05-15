"""
This is a test module for the CTM model.
"""
import logging
from rich.logging import RichHandler
from src.model.ctm import CTM
from src.common_utility.units import Units

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

def test_ctm_model():
    """
    Test the CTM model.
    """

    model = CTM.__new__(CTM)
    # model.dl = bypassed_data_loader
    # cell_length = bypassed_data_loader.geo_loader.cell_length
    # assert model.dl.params.get_spatial_line_capacity(cell_length) == 90
    # assert model.dl.params.flow_capacity == 15
    cell_occupancies = [27, 87, 3]
    dt = 30 * Units.S
    first_cell_inflow = (1440 * Units.PER_HR) * dt
    # 1440 * 1/60 = 24 veh/min
    cell_capacities = [
        90,
        90,
        90,
    ]
    alpha = 1
    q_max = 1800 * Units.PER_HR
    flow_capacity = q_max * dt
    max_flows = [flow_capacity for _ in range(len(cell_capacities))]
    values = model.run({
            "occupancy_list":cell_occupancies,
            "first_cell_inflow":first_cell_inflow,
            "next_occupancy": [0, 0, 0],
            "trajectory_time": 0,
            "link_id": 1,
            "is_tl":True,
            "alpha": alpha,
            "tl_status":False,
            "cell_capacities":cell_capacities,
            "flow_capacity":flow_capacity,
            "max_flows":max_flows
    }
    )
    # Adjust the expected value based on the actual output of model.predict
    assert values["new_occupancy"] == [36, 75, 15]

    values = model.run({
            "occupancy_list":values["new_occupancy"],
            "first_cell_inflow":first_cell_inflow,
            "next_occupancy": [0, 0, 0],
            "trajectory_time": 0,
            "link_id": 1,
            "is_tl":True,
            "alpha": alpha,
            "tl_status":False,
            "cell_capacities":cell_capacities,
            "flow_capacity":flow_capacity,
            "max_flows":max_flows
    }
    )
    # Adjust the expected value based on the actual output of model.predict
    assert values["new_occupancy"] == [33, 75, 15]

    values = model.run({
            "occupancy_list":values["new_occupancy"],
            "first_cell_inflow":first_cell_inflow,
            "next_occupancy": [0, 0, 0],
            "trajectory_time": 0,
            "link_id": 1,
            "is_tl":True,
            "alpha": alpha,
            "tl_status":False,
            "cell_capacities":cell_capacities,
            "flow_capacity":flow_capacity,
            "max_flows":max_flows
    }
    )
    # Adjust the expected value based on the actual output of model.predict
    assert values["new_occupancy"] == [30, 75, 15]
