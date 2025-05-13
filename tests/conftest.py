"""
Processes a given selection of data or code.

Performs operations such as parsing, analyzing, or transforming the input,
depending on the context.

Returns:
    The result of processing the selection.

Raises:
    Exception: If the selection is invalid or processing fails.
"""
from pathlib import Path
import polars as pl
import pytest

@pytest.fixture
def base_dir():
    """Fixture to provide the base directory of the project"""
    return Path(__file__).resolve().parent.parent

@pytest.fixture
def assets_path(base_dir):
    """
    Fixture to provide the path to the assets directory
    """
    assets_dir = base_dir / Path("tests/assets")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")
    return assets_dir

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