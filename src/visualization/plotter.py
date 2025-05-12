
"""
Plotter library for visualizing the data.
"""
import plotly.express as px
import json
import os
import polars as pl
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
        
        
        


    def plot(self):
        """
        Plotting the data
        """
        pass
