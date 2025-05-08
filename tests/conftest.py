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

@pytest.fixture
def sample_dataframe():
    """
    Creates a sample Polars DataFrame with 'timestamp' and 'value' columns.

    Returns:
        pl.DataFrame: A DataFrame containing example data with timestamps and corresponding values.
    """
    return pl.DataFrame({
        "timestamp": [0.0, 0.08, 0.12, 0.20, 0.48, 0.96],
        "value": [10, 20, 30, 40, 50, 60]
    })
