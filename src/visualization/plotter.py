
"""
Plotter library for visualizing the data.
"""
import json
import os
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
class Plotter:
    """
    A class for visualizing data using various plotting libraries.
    Attributes:
        data (Any): The data to be visualized.
    Methods:
        plot():
            Generates a plot of the data. This is a placeholder method and should be
            implemented using a plotting library such as matplotlib or seaborn.
    """
    def __init__(self, cache_dir: str):
        """
        Initializes the Plotter with the cache directory.

        Args:
            cache_dir (str): The directory where cached data is stored.
        """
        self.cache_dir = cache_dir

    def get_parameters(self, file_name: str):
        """
        Find parameters from the file name.

        Args:
            file_name (str): The name of the file to extract parameters from.

        Returns:
            dict: A dictionary containing the extracted parameters.
        """
        params_file_name = file_name.split("_")[-1]
        path_to_params_file = f"{self.cache_dir}/params/{params_file_name}"
        if os.path.exists(path_to_params_file):
            with open(path_to_params_file, "r") as f:
                params = json.load(f)
            return params
        else:
            raise FileNotFoundError(f"Parameters file not found: {path_to_params_file}")
        
    def get_rmse(self, file_name: str):
        """
        Find RMSE from the file name.

        Args:
            file_name (str): The name of the file to extract RMSE from.

        Returns:
            dict: A dictionary containing the extracted RMSE.
        """
        
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")\
        
        data = pl.read_json(file_name)
        rmse_data = data.group_by(["link_id", "trajectory_time"]).agg(
            (pl.col("new_occupancy") - pl.col("next_occupancy")).pow(2).mean().alias("rmse")
        )
        rmse_data_min = rmse_data["rmse"].min()
        rmse_data_max = rmse_data["rmse"].max()
        for name, group in rmse_data.group_by("link_id"):

            fig = go.Figure(data=go.Heatmap(
                z=group["rmse"],
                x=group["link_id"],
                y=group["trajectory_time"],
                colorscale="Reds",
                zmin=rmse_data_min,
                zmax=rmse_data_max
            ))
            fig.update_layout(
                title=f"Flow Actual Heatmap - Link ID {name}",  # Dynamic title per link_id
                xaxis_title="Cell Index",
                yaxis_title="Time",
                font=dict(size=14)  # Adjust font size for better readability
            )
            if not os.path.exists(f"{self.cache_dir}/heatmaps"):
                os.makedirs(f"{self.cache_dir}/heatmaps", exist_ok=True)
            fig.write_image(f"{self.cache_dir}/heatmaps/heatmap_link_{name[0]}.png")


    def plot(self):
        """
        Plotting the data
        """
        pass

if __name__ == "__main__":
    plotter = Plotter(cache_dir=".cache")
    file_name = ".cache/PointQueue/d1_20181029_0800_0830_682a48de_b23da350a0e6de66dcad3331001e8398.json"
    rmse = plotter.get_rmse(file_name)