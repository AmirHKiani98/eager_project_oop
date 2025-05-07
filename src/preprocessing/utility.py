"""
Utility functions for preprocessing data in the Eager Project.

This module provides helper functions to handle data preprocessing tasks,
such as filling missing timestamps in Polars DataFrames.

Functions:
    fill_missing_timestamps(group: pl.DataFrame) -> pl.DataFrame:
        Fills missing timestamps in a Polars DataFrame by creating placeholder rows
        with `None` values for missing data, ensuring a continuous sequence of
        timestamps with a fixed interval.

Dependencies:
    - polars
    - numpy
"""

import polars as pl
import numpy as np

def fill_missing_timestamps(group: pl.DataFrame, min_time: float, max_time: float, time_interval: float) -> pl.DataFrame:
    """
    Fills missing timestamps in a given Polars DataFrame by creating placeholder rows.

    This function ensures that the `trajectory_time` column in the input DataFrame
    contains a continuous sequence of timestamps with a fixed interval of 0.04 seconds.
    Missing timestamps are identified and placeholder rows are added with `None` values
    for all columns except `trajectory_time`, `link_id`, and `cell_id`, which are filled
    with the corresponding values from the first row of the input group.

    Args:
        group (pl.DataFrame): A Polars DataFrame containing at least the columns
            `trajectory_time`, `link_id`, and `cell_id`. The `trajectory_time` column
            must be numeric and represent timestamps.

    Returns:
        pl.DataFrame: A new Polars DataFrame with missing timestamps filled in. The
        resulting DataFrame is sorted by the `trajectory_time` column.

    Raises:
        ValueError: If the resulting DataFrame is empty after filling missing timestamps.

    Notes:
        - The function assumes that `trajectory_time` values are in ascending order
          within the input DataFrame.
        - Floating-point precision errors are mitigated by rounding timestamps to
          5 decimal places.
    """
    group = group.sort("trajectory_time")
    times = group["trajectory_time"].to_numpy()

    expected_times = np.arange(min_time, max_time + 0.001, time_interval).round(5)
    actual_times = set(np.round(times, 5))

    missing_times = [t for t in expected_times if t not in actual_times]

    if not missing_times:
        return group

    placeholders = {
        col: [None] * len(missing_times) for col in group.columns
    }
    placeholders["trajectory_time"] = missing_times
    placeholders["link_id"] = [group["link_id"][0]] * len(missing_times)
    placeholders["cell_id"] = [group["cell_id"][0]] * len(missing_times)

    placeholder_df = pl.DataFrame(placeholders)
    result_df = pl.concat([group, placeholder_df]).sort("trajectory_time")

    if result_df.is_empty():
        raise ValueError("Resulting DataFrame is empty after filling missing timestamps.")
    return result_df.sort("trajectory_time")
