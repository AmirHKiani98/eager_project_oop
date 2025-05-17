 
"""
Plotter library for visualizing the data.
"""
import json
import os
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
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
    def __init__(self, cache_dir: str, traffic_model: str = "CTM"):
        """
        Initializes the Plotter with the cache directory.

        Args:
            cache_dir (str): The directory where cached data is stored.
            traffic_model (str): The traffic model to be used for visualization.
        """
        self.cache_dir = cache_dir
        self.traffic_model = traffic_model
        os.makedirs(self.cache_dir + f"/results/{self.traffic_model}", exist_ok=True)

    def set_traffic_model_name(self, traffic_model: str):
        """
        Set the traffic model name.

        Args:
            traffic_model (str): The name of the traffic model.
        """
        self.traffic_model = traffic_model

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
        
    def plot_rmse_point_queue_spatial_queue(self, file_name: str):
        """
        Find RMSE from the file name.

        Args:
            file_name (str): The name of the file to extract RMSE from.
        """
        
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")\
        
        data = pl.read_json(file_name)
        rmse_data = data.group_by(["link_id", "trajectory_time"]).agg(
            ((pl.col("receiving_flow")-pl.col("sending_flow")) - pl.col("next_occupancy")).pow(1).mean().alias("rmse")
        )
        rmse_data = rmse_data.filter(
            # pl.col('rmse') < 20
        )
        
        folder_path = f"{self.cache_dir}/results/{self.traffic_model}/{self.get_base_name_without_extension(file_name)}"
        os.makedirs(folder_path, exist_ok=True)
        
        rmse_data_min = rmse_data["rmse"].min()
        rmse_data_max = rmse_data["rmse"].max()
        if not isinstance(rmse_data_min, float) or not isinstance(rmse_data_max, float):
            raise ValueError("RMSE data min and max should be float values.")
        import matplotlib.pyplot as plt

        # Convert Polars DataFrame to Pandas DataFrame for seaborn compatibility
        rmse_data_pd = rmse_data.to_pandas()

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
        folder_path = f"{self.cache_dir}/results/{self.traffic_model}/{self.get_base_name_without_extension(file_name)}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for name, group in groups:
            link_id = name[0]
            group = group.sort("trajectory_time")
            group = group.to_pandas().set_index("trajectory_time")

            plt.figure(figsize=(8, 6))
            sns.heatmap(group, annot=True, fmt=".1f", cmap="RdYlGn_r", cbar=False)
            plt.title(f"Heatmap for Link ID: {link_id}")
            plt.savefig(folder_path + f"/heatmap_link_{link_id}_{self.get_base_name_without_extension(file_name)}.png")
            plt.close()
    
    def plot_sns_heatmap_point_queue_spatial_queue(self, file_name: str):
        """
        Plotting the heatmap using seaborn.
        Args:
            file_name (str): The name of the file to plot.
        """
        data = pl.read_json(file_name)
        
        folder_path = f"{self.cache_dir}/results/{self.traffic_model}/{self.get_base_name_without_extension(file_name)}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
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
        plt.savefig("error_heatmap.png")
        
        
        # Set 't' as index
        


        # Plot
        # plt.figure(figsize=(8,6))
        # sns.heatmap(joined_group, annot=False, fmt=".1f", cmap="Reds", cbar=True)
        # plt.savefig("heatmap.png")
            
        



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
        unique_times = unique_times.tolist()
        print("Starting animation with ", len(unique_times), "frames.")
        unique_times = unique_times[::10]
        def update(frame_index):
            print(f"Frame {frame_index + 1}/{len(unique_times)}")
            current_time = unique_times[frame_index]
            frame = df[df["trajectory_time"] == current_time]
            x = frame["lon"]
            y = frame["lat"]
            scatter.set_offsets(np.c_[x, y])
            ax.set_title(f"Time: {current_time}")
            
            return scatter,
        ani = FuncAnimation(fig, update, frames=len(unique_times), blit=True)

        ani.save("animation.gif", writer='imagemagick', fps=10)
        

        print("Animation saved as 'animation.gif'.")


if __name__ == "__main__":
    plotter = Plotter(cache_dir=".cache")
    data_file_name = "d1_20181029_0800_0830"
    params_hash = "3a4a36a486cb5990adba742c60bc84ab"
    geo_hash = "682a48de"
    traffic_model_name = "PointQueue"
    try:
        plotter.animation(f".cache/{data_file_name}_fully_process_vehicles_{geo_hash}.csv")
        print("Heatmap generated and saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")