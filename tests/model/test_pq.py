"""
This is a test module for the Point Queue model.
"""
import logging
from rich.logging import RichHandler
from src.model.point_queue import PointQueue
from src.common_utility.units import Units

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

def test_point_queue():
    """
    Test the Point Queue model.
    Based on the table 9.1 - Stephen Boyles
    """
    pq = PointQueue.__new__(PointQueue)
    
    dt = 1 * Units.HR
    q_max_up = 10 * Units.PER_HR
    q_max_down = 5 * Units.PER_HR
    
    table = {
        "cumulative_count_upstream": [0, 1, 5, 10, 17, 27, 30, 30, 30, 30, 30],
        "cumulative_count_downstream": [0, 0, 0, 0, 1, 5, 10, 15, 20, 25, 30],
    }

    expected_values = {
        "sending_flow": [0, 0, 0, 1, 4, 5, 5, 5, 5, 5, 0],
        "receiving_flow": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    }
    for i in range(len(table["cumulative_count_upstream"])):
        cumulative_count_upstream = table["cumulative_count_upstream"][i]
        cumulative_count_downstream = table["cumulative_count_downstream"][i]
        
        values = pq.run({
            "cumulative_count_upstream": cumulative_count_upstream,
            "cumulative_count_downstream": cumulative_count_downstream,
            "dt": dt,
            "q_max_up": q_max_up,
            "q_max_down": q_max_down,
            "next_occupancy": 0
        })
        print(values, "\n", table["cumulative_count_upstream"][i], table["cumulative_count_downstream"][i])
        assert values["sending_flow"] == expected_values["sending_flow"][i]
        assert values["receiving_flow"] == expected_values["receiving_flow"][i]
