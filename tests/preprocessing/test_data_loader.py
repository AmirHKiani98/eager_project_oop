"""
"""
import polars as pl
from src.preprocessing.data_loader import DataLoader
def test_get_removed_vehicle_on_minor_roads_df(assets_path):
    """Test the get_removed_vehicle_on_minor_roads_df function."""
    

    # Test with a sample DataFrame
    dl = DataLoader.__new__(DataLoader)
    
    df = pl.read_csv(assets_path / "get_removed_vehicle_on_minor_roads_df.csv")

    # Call the function
    result_df = dl.get_removed_vehicle_on_minor_roads_df(df)
    # There should be only one track_id in the result and that should be 1
    # The length of the result should not be necessarily 1
    assert result_df["track_id"].unique().to_list() == [1], "track_id should only contain 1"
    assert len(result_df) > 0, "Resulting DataFrame should not be empty"
