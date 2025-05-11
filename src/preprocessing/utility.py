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

def fill_missing_timestamps(
    df: pl.DataFrame,
    column_name: str,
    interval: float = 0.04,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> pl.DataFrame:
    """
    Fills missing timestamps in a DataFrame by generating a complete range of values
    for a specified column.

    Parameters:
        df (pl.DataFrame): The input DataFrame containing the column with possible
            missing timestamps.
        column_name (str): The name of the column to fill missing timestamps for.
        interval (float, optional): The interval between consecutive timestamps.
            Defaults to 0.04.
        min_value (float, optional): The minimum value of the timestamp range.
            Defaults to 0.0.
        max_value (float, optional): The maximum value of the timestamp range.
            Defaults to 1.0.

    Returns:
        pl.DataFrame: A DataFrame with a complete range of timestamps in the specified
            column, with original data joined where available and missing rows filled
            with nulls.
    """

    column_values = np.arange(min_value, max_value + 0.001, interval)
    column_df = pl.DataFrame(
        {column_name: column_values}
    ).with_columns(
        pl.col(column_name).cast(pl.Float64)
    )
    full_range_df = column_df.join(df, on=column_name, how="left")
    return full_range_df
