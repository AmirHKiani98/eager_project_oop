"""
Module: test_utils
------------------
This module contains unit tests for utility functions used in the preprocessing
pipeline. Specifically, it tests the functionality of filling missing timestamps
in a Polars DataFrame.

Functions:
----------
- test_fill_missing_timestamps(sample_dataframe): 
    Tests the 'fill_missing_timestamps' function to ensure it correctly fills
    missing timestamps in a DataFrame, producing the expected output.
"""
import polars as pl

from src.preprocessing.utility import fill_missing_timestamps

def test_fill_missing_timestamps(sample_dataframe):
    """
    Test the fill_missing_timestamps function to ensure it correctly fills missing
    timestamps in a DataFrame.
    """

    expected_data = {
        "timestamp": [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
        "a": [10, None, 20, 30, None, 40, None, 50, 60],
        "b": [1, None, 2, 3, None, 4, None, 5, 6],
        "c": [0.1, None, 0.2, 0.3, None, 0.4, None, 0.5, 0.6],
    }
    expected_df = pl.DataFrame(expected_data)
    min_sdf = sample_dataframe["timestamp"].min()
    max_sdf = sample_dataframe["timestamp"].max()

    result_df = fill_missing_timestamps(sample_dataframe, "timestamp", 0.04, min_sdf, max_sdf)

    assert result_df.equals(expected_df)
