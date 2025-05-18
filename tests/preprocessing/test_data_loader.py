"""
Test the prepare_explode_dataset method of the DataLoader class.
This test bypasses the DataLoader's __init__ method, manually sets the geo_loader attribute,
and verifies that the prepare_explode_dataset method returns a DataFrame with the expected
structure and values.
    simple_geo_loader: A simple geo_loader object to be assigned to the DataLoader instance.
Asserts:
    - The result is an instance of pl.DataFrame.
    - The result has the expected columns and values as defined in expected_df.
    - The shape and content of the result match the expected DataFrame.
"""
import logging
import json
import polars as pl
from polars.testing import assert_frame_equal
from rich.logging import RichHandler
import numpy as np
from shapely.geometry import Point as POINT
from src.preprocessing.data_loader import DataLoader
from src.common_utility.units import Units
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

def test_get_trajectory_dataframe(sample_test_pneuma_dataframe_path, simple_geo_loader):
    """
    Test the _get_trajectory_dataframe function.

    Args:
        sample_test_pneuma_dataframe_path (pl.DataFrame): Sample DataFrame for testing.
    """
    # Bypassing the __init__ method of DataLoader
    dl = DataLoader.__new__(DataLoader)
    dl.geo_loader = simple_geo_loader
    # Call the function with the sample DataFrame
    result = dl.prepare_explode_dataset(sample_test_pneuma_dataframe_path)
    assert isinstance(result, pl.DataFrame)


    # Check if the result has the expected columns
    expected_df = {
        "track_id": [1, 1],
        "veh_type": ["Motorcycle", "Motorcycle"],
        "traveled_d": [416.10, 416.10],
        "avg_speed": [35.163320, 35.163320],
        "lat": [37.980090, 37.980088],
        "lon": [23.735504, 23.735506],
        "speed": [29.0144, 29.0140],
        "lon_acc": [-0.0072, 0.0017],
        "lat_acc": [-0.2004, -0.2314],
        "trajectory_time": [0.000000, 0.040000]
    }
    expected_df = pl.DataFrame(expected_df)
    for col in expected_df.columns:
        result = result.with_columns(
            pl.col(col).cast(expected_df[col].dtype)
        )
    assert expected_df.shape == result.shape
    assert_frame_equal(result, expected_df, check_column_order=False)


def test_is_vehicle_passed_traffic_light():
    """
    Test the is_vehicle_passed_traffic_light function.

    Args:
        sample_test_pneuma_dataframe_path (pl.DataFrame): Sample DataFrame for testing.
    """
    # Bypassing the __init__ method of DataLoader
    dl = DataLoader.__new__(DataLoader)

    veh_before_traffic_light = POINT(23.735259282298564, 37.980360795602905)
    intersection = POINT(23.7353800402736, 37.98021357743111)
    veh_after_traffic_light = POINT(23.73546054559029, 37.98010443273381)
    assert not dl.is_vehicle_passed_traffic_light(veh_before_traffic_light, intersection)
    assert dl.is_vehicle_passed_traffic_light(veh_after_traffic_light, intersection)


def test_density_exit_entered(sample_fully_modified_dataframe_path, simple_geo_loader):
    """
    Test the density_exit_entered function.

    Args:
        sample_fully_modified_dataframe_path (pl.DataFrame): Sample DataFrame for testing.
    """
    # Bypassing the __init__ method of DataLoader
    dl = DataLoader.__new__(DataLoader)
    dl.geo_loader = simple_geo_loader
    dl.time_interval = 0.04
    # Call the function with the sample DataFrame

    result = dl.get_density_entry_exit_df(sample_fully_modified_dataframe_path)
    cell_length = simple_geo_loader.cell_length
    # Check density at time 0, link 5, cell 2
    density_row = result.filter(
        (pl.col("trajectory_time") == 0) & (pl.col("link_id") == 5) & (pl.col("cell_id") == 2)
    )
    assert density_row.shape[0] == 1
    expected_density = 3 / cell_length.to(Units.M).value
    # Check if the density is within a small tolerance of the expected value
    assert abs(density_row["density"][0] - expected_density) < 1e-6
    link5_cell2 = result.filter(
        (pl.col("link_id") == 5) & (pl.col("cell_id") == 2)
    )
    link2_cell2 = result.filter(
        (pl.col("link_id") == 2) & (pl.col("cell_id") == 2)
    )
    link2_cell1 = result.filter(
        (pl.col("link_id") == 2) & (pl.col("cell_id") == 1)
    )

    # Check if the time intervals for all rows/links are consistent
    pl.testing.assert_series_equal(
        link5_cell2["trajectory_time"], link2_cell1["trajectory_time"], check_dtypes=True,
        check_order=False
    )
    pl.testing.assert_series_equal(
        link5_cell2["trajectory_time"], link2_cell2["trajectory_time"], check_dtypes=True,
        check_order=False
    )
    pl.testing.assert_series_equal(
        link2_cell2["trajectory_time"], link2_cell1["trajectory_time"], check_dtypes=True,
        check_order=False
    )


def test_cumulative_df(base_dir):
    """
    Test the cumulative_df function.
    """
    df = pl.read_csv(
        base_dir / "tests" / "assets" / "cummulative_df.csv",
    )
    # Bypassing the __init__ method of DataLoader
    dl = DataLoader.__new__(DataLoader)
    dt = -2 * Units.S
    dataframe = {
        "trajectory_time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "link_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "cumulative_link_entry": [0, 1, 5, 10, 17, 27, 30, 30, 30, 30, 30],
        "cumulative_link_exit": [0, 0, 0, 0, 1, 5, 10, 15, 20, 25, 30],
        "first_cell_entry": [0, 1, 5, 10, 17, 27, 30, 30, 30, 30, 30],
        "current_number_of_vehicles": [0, 0, 0, 0, 1, 5, 10, 15, 20, 25, 30],
    }
    df = pl.DataFrame(dataframe)
    results = dl.get_cummulative_counts_based_on_t(df, dt)
    print(results.sort("trajectory_time").select(
        pl.col("cummulative_count_upstream_offset"),
        pl.col("cummulative_count_downstream"),
    ).to_numpy())
    cummulative_count_upstream_offset = results.select(
        pl.col("cummulative_count_upstream_offset")
    ).to_numpy()
    cummulative_count_downstream = results.select(
        pl.col("cummulative_count_downstream")
    ).to_numpy()
    expected_cummulative_count_upstream_offset = [0, 0, 0, 1, 5, 10, 17, 27, 30, 30, 30]
    expected_cummulative_count_downstream = [0, 0, 0, 0, 1, 5, 10, 15, 20, 25, 30]
    assert list(cummulative_count_upstream_offset) == list(expected_cummulative_count_upstream_offset)
    assert list(cummulative_count_downstream) == list(expected_cummulative_count_downstream)

def test_density_entry_exit():
    """
    Test the density_entry_exit function.
    """

    dl = DataLoader.__new__(DataLoader)
    