 
"""
Plotter library for visualizing the data.
"""
import json
import os
from typing import Optional
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import pandas as pd
from tqdm import tqdm
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
        self.cell_length = {
            link_id: {
                cell_id: cell.get_length().to(Units.M).value for cell_id, cell in link.cells.items()
            } for link_id, link in self.geo_loader.links.items()
        }

        self.errors = {}
        all_errors_path = f"{self.cache_dir}/all_errors.json"
        if os.path.exists(all_errors_path):
            with open(all_errors_path, "r") as f:
                self.errors = json.load(f)
        
        

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
            traffic_model: str,
            params: Optional[tuple] = None):
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
        print(file_name)
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
        average_error = rmse_data["rmse"].mean()
        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error
            self.save_errors()
        rmse_data = rmse_data.filter(
            # pl.col('rmse') < 20
        )
        
        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/error.png"
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        
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
            cbar_kws={'label': 'Error'}
        )

        plt.title("Flow Actual Heatmap")
        plt.xlabel("Link Index")
        plt.ylabel("Time")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(figure_path)
        plt.close()  

    def truncate_and_square_error_ctm(self, row):
        new_occ = np.array(row["new_occupancy"])
        next_occ = np.array(row["next_occupancy"])
        lengths = np.array(row["cell_lengths"])

        min_len = min(len(new_occ), len(next_occ), len(lengths))
        return ((new_occ[:min_len] - next_occ[:min_len]) / lengths[:min_len]) ** 2

    def plot_error_ctm(self,
                       data_file_name: str,
            hash_parmas: str,
            hash_geo: str,
            traffic_model: str, params: Optional[tuple] = None):
        """
        Plots the error in CTM.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_parmas (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
        """
        average_error = 0
        n = 0
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_parmas}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        data = pl.read_json(
            file_name
        )
        data = data.with_columns(
            pl.col("link_id")
            .cast(pl.Int64)
            .map_elements(lambda link_id: list(self.cell_length.get(link_id, {}).values()))
            .alias("cell_lengths")
        )

        data = data.with_columns(
            pl.struct(["new_occupancy", "next_occupancy", "cell_lengths"])
            .map_elements(self.truncate_and_square_error_ctm)
            .alias("squared_error")
        )
        error_info = data.select(
            ["link_id", "trajectory_time", "squared_error"]
        )
        groups = error_info.group_by(["link_id"])
        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        all_errors = error_info.to_pandas()["squared_error"].explode().astype(float)
        rmse_data_min = all_errors.min()
        rmse_data_max = all_errors.max()
        for name, group in groups:
            link_id = int(name[0]) # type: ignore
            group = group.sort("trajectory_time")
            rmse_data = {
                "trajectory_time": [],
                "squared_error": [],
                "cell_id": [],
            }
            for row in tqdm(group.iter_rows(named=True), desc=f"Processing link {link_id}", total=len(group)):
                trajectory_time = row["trajectory_time"]
                squared_error = row["squared_error"]
                for i in range(len(squared_error)):
                    rmse_data["trajectory_time"].append(trajectory_time)
                    rmse_data["squared_error"].append(squared_error[i])
                    average_error += squared_error[i]
                    n += 1
                    rmse_data["cell_id"].append(i)
            rmse_data = pd.DataFrame(rmse_data)
            heatmap_data = rmse_data.pivot(index="trajectory_time", columns="cell_id", values="squared_error")
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                heatmap_data,
                cmap="Reds",
                vmin=rmse_data_min,
                vmax=rmse_data_max,
                annot=False,
                cbar_kws={'label': 'Error'}
            )
            if params is not None:
                if traffic_model not in self.errors:
                    self.errors[traffic_model] = {}
                
                str_key = str(params)
                self.errors[traffic_model][str_key] = average_error/n
                self.save_errors()
            plt.title(f"Heatmap for Link ID: {link_id}")
            plt.savefig(figure_path + f"Link_{link_id}.png")
            plt.close()
        # data = data.with_columns(
        #     pl.struct(["new_outflow", "next_occupancy", "cell_lengths"])
        #     .map_elements(lambda row: ((np.array(row["new_outflow"]) - np.array(row["next_occupancy"])) / np.array(row["cell_lengths"]))**2)
        #     .alias("squared_error")
        # )
        # print(data.select(["squared_error"]).head(10))


    def plot_sns_heatmap_ctm(self, 
                             file_name: str,
                            traffic_model: str):
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
            sns.heatmap(group, annot=False, fmt=".1f", cmap="RdYlGn_r", cbar=True)
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
            
        
    def plot_error_ltm(self,
                       data_file_name,
                       hash_parmas: str,
                       hash_geo: str,
                       traffic_model: str,
                       params: Optional[tuple] = None):
        """
        Plotting the error in LTM.
        Args:
            data_file_name (str): The name of the file to plot.
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
            pl.struct(["link_id", "cell_id"])
            .map_elements(lambda row: self.cell_length[int(row["link_id"])][int(row["cell_id"])], return_dtype=pl.Float64)
            .alias("cell_length")
        )

        data = data.with_columns(
            ((pl.col("new_occupancy") - pl.col("next_occupancy")) /
            pl.col("cell_length")).pow(2).alias("squared_error")
        )
        data = data.with_columns(
            pl.col("x").map_elements(lambda x: round(x, 2)
                                     ).alias("x")                                     
        )
        error_info = data.select(
            ["link_id", "trajectory_time", "squared_error", "x"]
        )
        average_error = data["squared_error"].mean()
        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error
            self.save_errors()
        groups = error_info.group_by(["link_id"])
        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        min_errors = error_info["squared_error"].cast(pl.Float64).min()
        max_errors = error_info["squared_error"].max()
        for name, group in groups:
            link_id = int(name[0]) # type: ignore
            group = group.sort("trajectory_time")
            group = group.to_pandas()
            rmse_data = group.pivot(
                index="trajectory_time", columns="x", values="squared_error"
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                rmse_data,
                cmap="Reds",
                vmin=min_errors, # type: ignore
                vmax=max_errors, # type: ignore
                annot=False,
                cbar_kws={'label': 'Error'}
            )
            plt.title(f"Heatmap for Link ID: {link_id}")
            plt.xlabel("X")
            plt.ylabel("Time")
            plt.tight_layout()
            plt.savefig(figure_path + f"Link_{link_id}.png")
            plt.close()








    def get_base_name_without_extension(self, file_name: str):
        """
        Get the base name of the file without extension.

        Args:
            file_name (str): The name of the file.

        Returns:
            str: The base name of the file without extension.
        """
        return os.path.splitext(os.path.basename(file_name))[0]

    def plot_error_pw(
            self,
            data_file_name: str,
            hash_parmas: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None
    ):
        """
        data_file_name (str): The name of the file to plot.
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
        data = data.filter(
            pl.col("link_id") != 5
        )
        data = data.with_columns(
            pl.struct(["new_densities", "next_densities"])
            .map_elements(lambda row: ((np.array(row["new_densities"]) - np.array(row["next_densities"])) / len(row["new_densities"]))**2)
            .alias("squared_error")
        )
        print(data.select(["squared_error"]).head(10))
        exit()
        

    def plot(self,
             data_file_name: str,
            hash_parmas: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None):
        """
        Plotting the data
        """
        if traffic_model == "LTM":
            self.plot_error_ltm(
                data_file_name=data_file_name,
                hash_parmas=hash_parmas,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
        elif traffic_model == "CTM":
            self.plot_error_ctm(
                data_file_name=data_file_name,
                hash_parmas=hash_parmas,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
        elif traffic_model == "PointQueue" or traffic_model == "SpatialQueue":
            self.plot_error_point_queue_spatial_queue(
                data_file_name=data_file_name,
                hash_parmas=hash_parmas,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
        elif traffic_model == "PW":
            self.plot_error_pw(
                data_file_name=data_file_name,
                hash_parmas=hash_parmas,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
        else:
            raise ValueError(f"Traffic model {traffic_model} not supported")
    
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

    def save_errors(self):
        """
        Save the errors.
        """
        all_errors_path = f"{self.cache_dir}/all_errors.json"
        with open(all_errors_path, "w") as f:
            json.dump(self.errors, f, indent=4)

    def load_errors(self):
        """
        Load the errors from a JSON file.
        """
        all_errors_path = f"{self.cache_dir}/all_errors.json"
        with open(all_errors_path, "r") as f:
            self.errors = json.load(f)

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
    traffic_model_name = "LTM"
    plotter = Plotter(cache_dir=".cache", geo_loader=model_geo_loader)
    # plotter.animation(f".cache/{data_file_name}_fully_process_vehicles_{geo_hash}.csv")
    # print("Heatmap generated and saved successfully.")
    plotter.plot_error_ltm(
        data_file_name=data_file_name,
        hash_parmas=params_hash,
        hash_geo=geo_hash,
        traffic_model=traffic_model_name
    )
