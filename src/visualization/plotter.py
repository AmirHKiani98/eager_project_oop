 
"""
Plotter library for visualizing the data.
"""
import json
import os
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import pandas as pd
import numpy as np
from shapely.geometry import Point as POINT
from src.preprocessing.geo_loader import GeoLoader
from src.common_utility.units import Units
matplotlib.set_loglevel("warning")
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
    def __init__(self, cache_dir: str, geo_loader: GeoLoader):
        """
        Initializes the Plotter with the cache directory.

        Args:
            cache_dir (str): The directory where cached data is stored.
            traffic_model (str): The traffic model to be used for visualization.
        """
        self.cache_dir = cache_dir
        self.geo_loader = geo_loader
        self.link_lengths = {
            link_id: link.get_length().to(Units.M).value
            for link_id, link in self.geo_loader.links.items()
        }
        

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
        
    def plot_error_point_queue_spatial_queue(self,
                                             data_file_name: str,
            hash_parmas: str,
            hash_geo: str,
            traffic_model: str):
        """
        Find RMSE from the file name.

        Args:
            data_file_name (str): The name of the file to extract parameters from.
            hash_parmas (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
        """
        
        
        
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_parmas}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        data = pl.read_json(
            file_name
        )
        
        data = data.with_columns(
            pl.col("link_id").cast(pl.Int64).replace(self.link_lengths).alias("link_length")
        )
        rmse_data = data.group_by(["link_id", "trajectory_time"]).agg(
            (((pl.col("receiving_flow") - pl.col("outflow")) - pl.col("next_occupancy")) /
             (pl.col("link_length"))).pow(2).mean().alias("rmse")
        )
        rmse_data = rmse_data.filter(
            # pl.col('rmse') < 20
        )
        
        folder_path = f"{self.cache_dir}/results/{traffic_model}/{self.get_base_name_without_extension(file_name)}"
        os.makedirs(folder_path, exist_ok=True)
        
        rmse_data_min = rmse_data["rmse"].min()
        rmse_data_max = rmse_data["rmse"].max()
        if not isinstance(rmse_data_min, float) or not isinstance(rmse_data_max, float):
            raise ValueError("RMSE data min and max should be float values.")
        import matplotlib.pyplot as plt

        # Convert Polars DataFrame to Pandas DataFrame for seaborn compatibility
        rmse_data_pd = rmse_data.to_pandas()
        rmse_data_pd["link_id"] = rmse_data_pd["link_id"].astype(int)
        # Pivot the data for heatmap
        heatmap_data = rmse_data_pd.pivot(index="trajectory_time", columns="link_id", values="rmse")

        # Plot the heatmap using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            cmap="Reds",
            vmin=rmse_data_min,
            vmax=rmse_data_max,
            annot=False,
            cbar_kws={'label': 'RMSE'}
        )

        plt.title("Flow Actual Heatmap")
        plt.xlabel("Link Index")
        plt.ylabel("Time")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(f"{folder_path}/heatmap.png")
        plt.close()  

    def plot_sns_heatmap_ctm(self, file_name: str):
        """
        Plotting the heatmap using seaborn.

        Args:
            file_name (str): The name of the file to plot.
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        
        data = pl.read_json(file_name)
        groups = data.group_by(["link_id"])
        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}.png"
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        for name, group in groups:
            link_id = name[0]
            group = group.sort("trajectory_time")
            group = group.to_pandas().set_index("trajectory_time")

            plt.figure(figsize=(8, 6))
            sns.heatmap(group, annot=True, fmt=".1f", cmap="RdYlGn_r", cbar=False)
            plt.title(f"Heatmap for Link ID: {link_id}")
            plt.savefig(figure_path)
            plt.close()
    
    def plot_sns_heatmap_point_queue_spatial_queue(self,
                                                   data_file_name: str,
                                                   hash_parmas: str,
                                                   hash_geo: str,
                                                   traffic_model: str):
        """
        Plotting the heatmap using seaborn.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_parmas (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
        """
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_parmas}.json"
        data = pl.read_json(
            file_name
        )
        
        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}.png"
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        
        data = data.with_columns(
            ((pl.col("receiving_flow") - pl.col("sending_flow")) - pl.col("next_occupancy")).alias("error")
        )
        data = data.filter(
            pl.col("error") > 0
        )
        print(data["sending_flow"].mean())
        df_pd = data.to_pandas()
        df_pd["link_id"] = df_pd["link_id"].astype(int)

        # Pivot the table: rows = time, columns = links, values = error
        heatmap_data = df_pd.pivot(index="trajectory_time", columns="link_id", values="error")
        
        # Plot with Seaborn

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            cmap="Reds",
            cbar=True,
            cbar_kws={"label": "Error"},
            linewidths=1
        )
        plt.title("Error Heatmap by Link and Trajectory Time")
        plt.xlabel("Link ID")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.savefig(figure_path)
            
        



    def get_base_name_without_extension(self, file_name: str):
        """
        Get the base name of the file without extension.

        Args:
            file_name (str): The name of the file.

        Returns:
            str: The base name of the file without extension.
        """
        return os.path.splitext(os.path.basename(file_name))[0]

    def plot(self):
        """
        Plotting the data
        """
        pass
    
    def animation(self, file_name: str):
        """
        Animation of the data
        """
        df = pd.read_csv(file_name)
        df = df[df["track_id"] == 70]
        fig, ax = plt.subplots()
        scatter = ax.scatter([], [])
        x_min, x_max = df["lon"].min(), df["lon"].max()
        y_min, y_max = df["lat"].min(), df["lat"].max()
        if x_min == x_max:
            x_min -= 0.0001
            x_max += 0.0001
        if y_min == y_max:
            y_min -= 0.0001
            y_max += 0.0001

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        unique_times = df["trajectory_time"].unique()
        # sort the times
        unique_times.sort()
        unique_times = unique_times[(unique_times>40) & (unique_times<60)]
        unique_times = unique_times.tolist()
        print("Starting animation with ", len(unique_times), "frames.")
        unique_times = unique_times[::1]
        def update(frame_index):
            current_time = unique_times[frame_index]
            frame = df[df["trajectory_time"] == current_time]
            print(frame["link_id"].values[0], frame["cell_id"].values[0], frame["lon"].values[0], frame["lat"].values[0], current_time)  
            x = frame["lon"]
            y = frame["lat"]
            scatter.set_offsets(np.c_[x, y])
            ax.set_title(f"Time: {current_time}")
            
            return scatter,
        ani = FuncAnimation(fig, update, frames=len(unique_times), blit=True)

        ani.save("animation.gif", writer='imagemagick', fps=5)
        print("Animation saved as 'animation.gif'.")


if __name__ == "__main__":
    intersection_locations = (
        pl.read_csv(".cache/traffic_lights.csv")
        .to_numpy()
        .tolist()
    )
    intersection_locations = [
        POINT(loc[1], loc[0])
        for loc in intersection_locations
    ]
    model_geo_loader = GeoLoader(
        locations=intersection_locations,
        cell_length=20.0
        )
    data_file_name = "d1_20181029_0800_0830"
    params_hash = "dcca17e9025816395dbe6a5a465c2450"
    geo_hash = "682a48de"
    traffic_model_name = "PointQueue"
    plotter = Plotter(cache_dir=".cache", geo_loader=model_geo_loader)
    # plotter.animation(f".cache/{data_file_name}_fully_process_vehicles_{geo_hash}.csv")
    # print("Heatmap generated and saved successfully.")
    plotter.plot_error_point_queue_spatial_queue(
        data_file_name=data_file_name,
        hash_parmas=params_hash,
        hash_geo=geo_hash,
        traffic_model=traffic_model_name
    )
