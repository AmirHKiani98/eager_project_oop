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
import polars as pl
from polars.testing import assert_frame_equal
from shapely.geometry import Point as POINT
from src.preprocessing.data_loader import DataLoader
def test__get_trajectory_dataframe(sample_test_pneuma_dataframe_path, simple_geo_loader):
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
    result = dl.get_density_entry_exist_df(sample_fully_modified_dataframe_path)
    cell_length = simple_geo_loader.cell_length
    # Check density at time 0, link 5, cell 2
    density_row = result.filter(
        (pl.col("trajectory_time") == 0) & (pl.col("link_id") == 5) & (pl.col("cell_id") == 2)
    )
    assert density_row.shape[0] == 1
    expected_density = 3 / cell_length
    assert abs(density_row["density"][0] - expected_density) < 1e-6
