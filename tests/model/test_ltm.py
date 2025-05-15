"""
Tests for the `LTM` model.
"""
import polars as pl
from src.model.ltm import LTM
from src.preprocessing.data_loader import DataLoader
from src.common_utility.units import Units
def test_ltm_model():
    """
    Test based on Table 9.6 of the book "Transportation Network Analysis"
    by Stephen Boyles.
    """
    table = {
        "t": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "d(t)": [10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        "R(t)": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 1, 0, 0, 10, 10, 10, 10, 10, 10, 10],
        "Inflow": [10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 1, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        "N↑(t)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        "N↓(t)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        "S(t)": [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 0],
        "Outflow": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 0],
        "Vehicles on link": [0, 10, 20, 30, 40, 50, 59, 67, 74, 80, 85, 79, 70, 60, 50, 45, 35, 25, 15, 5, 0],
    }
    tl_status = [0 if t <= 10 else 1 for t in range(20)]
    # Create a DataFrame from the table
    df = pl.DataFrame(table)
    dt_sending_flow = -2 * Units.HR
    dt_receiving_flow = -3 * Units.HR
    q_max = 10 * Units.PER_HR
    dt = 1 * Units.HR
    k_j = 180 * Units.PER_KM
    link_length = 0.5 * Units.KM
    for row in df.iter_rows(named=True):
        t = row["t"] * Units.HR
        target_t_sending_flow = t + dt_sending_flow
        target_t_receiving_flow = t + dt_receiving_flow
        
        if target_t_sending_flow < 0:
            cummulative_upstream_offset = 0
        else:
            cummulative_upstream_offset = df.filter(pl.col("t") == target_t_sending_flow)["N↑(t)"].to_numpy()[0]
        if target_t_receiving_flow < 0:
            cummulative_downstream_offset = 0
        else:
            cummulative_downstream_offset = df.filter(pl.col("t") == target_t_receiving_flow)["N↓(t)"].to_numpy()[0]
        cummulative_upstream = row["N↑(t)"]
        cummulative_downstream = row["N↓(t)"]
        entry_count = row["Inflow"]
        current_number_of_vehicles = row["Vehicles on link"]
        result = LTM.run({
            "q_max": q_max, 
            "next_occupancy": 0, 
            "cummulative_count_upstream_sending_flow": cummulative_upstream_offset, 
            "cummulative_count_upstream": cummulative_upstream, 
            "cummulative_count_downstream": cummulative_downstream, 
            "cummulative_count_downstream_receiving_flow": cummulative_downstream_offset,
            "dt": dt,
            "link_length": link_length,
            "k_j": k_j,
            "trajectory_time": 0,
            "link_id": 0,
            "entry_count": entry_count,
            "current_number_of_vehicles": current_number_of_vehicles
        })
        

    assert False
