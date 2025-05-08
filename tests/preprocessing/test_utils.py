import polars as pl
import numpy as np
import pytest

from src.preprocessing.utility import fill_missing_timestamps

def test_fill_missing_timestamps_fills_gaps():
    # Create a DataFrame with missing timestamps
    df = pl.DataFrame({
        "timestamp": [0.0, 0.08, 0.12],  # Missing 0.04
        "value": [10, 20, 30]
    })

    # Call the function
    result = fill_missing_timestamps(
        df, 
        column_name="timestamp", 
        interval=0.04, 
        min_value=0.0, 
        max_value=0.12
    )

    # Expected timestamps: 0.00, 0.04, 0.08, 0.12
    expected_timestamps = np.array([0.00, 0.04, 0.08, 0.12])
    assert np.allclose(result["timestamp"].to_numpy(), expected_timestamps)

    # Check that the missing row (0.04) has null in 'value'
    value_list = result["value"].to_list()
    assert value_list[1] is None  # 0.04 row should be None
    assert value_list[0] == 10
    assert value_list[2] == 20
    assert value_list[3] == 30