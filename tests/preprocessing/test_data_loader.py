from src.preprocessing.data_loader import DataLoader
import polars as pl
def test__get_trajectory_dataframe(sample_test_pneuma_dataframe_path, simple_geo_loader):
    """
    Test the _get_trajectory_dataframe function.

    Args:
        sample_test_pneuma_dataframe_path (pl.DataFrame): Sample DataFrame for testing.
    """
    # Bypassing the __init__ method of DataLoader
    dl = DataLoader.__new__(DataLoader)

    # Call the function with the sample DataFrame
    result = dl.prepare_explode_dataset(sample_test_pneuma_dataframe_path)

    # Check if the result is a DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check if the result has the expected columns
    expeceted_data = {
        "track_id": [1, 1],
        "type": ["Motorcycle", "Motorcycle"],
        "traveled_d": [41, 6.10, 416.10],
        "avg_speed": [37.980090, 37.980090],
        "lat": [37.980090, 37.980088],
        "lon": [23.735504, 23.735506],
        "speed": [29.0144, 29.0140],
        "lon_acc": [-0.0072, 0.0017],
        "lat_acc": [-0.2004, -0.2314],
        "time": [0.000000, 0.040000]
    }
    excpected_df = pl.DataFrame(expeceted_data)
    
    assert excpected_df.shape == result.shape
    assert excpected_df.equals(result)