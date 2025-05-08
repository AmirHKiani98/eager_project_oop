"""
Processes a given selection of data or code.

Performs operations such as parsing, analyzing, or transforming the input,
depending on the context.

Returns:
    The result of processing the selection.

Raises:
    Exception: If the selection is invalid or processing fails.
"""
import polars as pl
import pytest
from shapely.geometry import Point as POINT
from src.preprocessing.geo_loader import GeoLoader
from src.preprocessing.data_loader import DataLoader
from src.model.params import Parameters

@pytest.fixture
def simple_traffic_params():
    """
    Creates and returns a Parameters instance with bypassed parameters.

    Returns:
        Parameters: An instance of Parameters with bypassed parameters.
    """
    return Parameters(free_flow_speed=50.0, dt=30.0, jam_density_link=180.0, q_max=3000.0)

@pytest.fixture
def bypassed_data_loader():
    """
    Creates and returns a DataLoader instance with bypassed data loading.

    Returns:
        DataLoader: An instance of DataLoader with bypassed data loading.
    """
    return DataLoader.__new__(DataLoader)

@pytest.fixture
def simple_geo_loader():
    """
    Creates and returns a GeoLoader instance initialized with two POINT locations
    and a cell length of 20.0.

    Returns:
        GeoLoader: An instance of GeoLoader with predefined locations and cell length.
    """
    # Example usage
    intersection_locations = (
        pl.read_csv(".cache/traffic_lights.csv")
        .to_numpy()
        .tolist()   # It's format is [lat, lon]
    )
    intersection_locations = [
        POINT(loc[1], loc[0])
        for loc in intersection_locations
    ]  # It's format is [lat, lon]
    model_geo_loader = GeoLoader(
        locations=intersection_locations,
        cell_length=20.0
    )
    return model_geo_loader

@pytest.fixture
def sample_dataframe():
    """
    Creates a sample Polars DataFrame with 'timestamp' and 'value' columns.

    Returns:
        pl.DataFrame: A DataFrame containing example data with timestamps and corresponding values.
    """
    return pl.DataFrame({
        "timestamp": [0.0, 0.08, 0.12, 0.20, 0.28, 0.32],
        "a": [10, 20, 30, 40, 50, 60],
        "b": [1, 2, 3, 4, 5, 6],
        "c": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })

@pytest.fixture
def sample_test_pneuma_dataframe_path():
    """
    Creates a sample Polars DataFrame with 'timestamp' and 'traffic' columns.

    Returns:
        str: The path to a CSV file containing
            example data with timestamps and corresponding traffic values.
    """
    return "tests/assets/test_pneuma_df.csv"


@pytest.fixture
def sample_fully_modified_dataframe_path():
    """
    Creates a sample Polars DataFrame with 'timestamp' and 'traffic' columns.
    Returns:
        str: The path to a CSV file containing
            example data with timestamps and corresponding traffic values.
    """
    return "tests/assets/test_fully_modified_data_df.csv"