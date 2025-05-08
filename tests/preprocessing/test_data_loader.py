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
    