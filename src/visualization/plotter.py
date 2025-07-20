 
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
from collections import defaultdict
from pyproj import Geod
import matplotlib.cm as cm
color_map = plt.get_cmap('tab20')

GEOD = Geod(ellps="WGS84")
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
    def __init__(self, cache_dir: str):
        """
        Initializes the Plotter with the cache directory.

        Args:
            cache_dir (str): The directory where cached data is stored.
            traffic_model (str): The traffic model to be used for visualization.
        """
        self.cache_dir = cache_dir
        

        self.errors = {}
        all_errors_path = f"{self.cache_dir}/all_errors.json"
        if os.path.exists(all_errors_path):
            with open(all_errors_path, "r") as f:
                self.errors = json.load(f)
    
    def _get_geo_loader(self, geo_loader_hash: str):

        return GeoLoader(
            locations=None,
            cache=self.cache_dir,
            hash_str=geo_loader_hash
        )

    def _get_link_lengths(self, geo_loader_hash: str) -> dict:
        """
        Get the link lengths from the geo loader.
        Args:
            geo_loader_hash (str): The hash of the geo loader.
        Returns:
            dict: A dictionary mapping link IDs to their lengths.
        """
        geo_loader = self._get_geo_loader(geo_loader_hash)
        return {
            link_id: link.get_length().to(Units.M).value
            for link_id, link in geo_loader.links.items()
        }
    def _get_cell_length(self, geo_loader_hash: str) -> dict:
        """
        Get the cell lengths from the geo loader.
        Args:
            geo_loader_hash (str): The hash of the geo loader.
        Returns:
            dict: A dictionary mapping link IDs to their cell lengths.
        """
        geo_loader = self._get_geo_loader(geo_loader_hash)
        return {
            link_id: {
                cell_id: cell.get_length().to(Units.M).value for cell_id, cell in link.cells.items()
            } for link_id, link in geo_loader.links.items()
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
            hash_params: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None):
        """
        Find RMSE from the file name.

        Args:
            data_file_name (str): The name of the file to extract parameters from.
            hash_params (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
        """

        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_params}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        
        data = pl.read_json(
            file_name
        )
        data = data.filter(
            pl.col("link_id") != 5
        )
        link_lengths = self._get_link_lengths(hash_geo)
        if not link_lengths:
            raise ValueError(f"No link lengths found for geo hash: {hash_geo}")
        data = data.with_columns(
            pl.col("link_id").cast(pl.Int64).replace(link_lengths).alias("link_length")
        )
        rmse_data = data.group_by(["link_id", "trajectory_time"]).agg(
            (((pl.col("receiving_flow") - pl.col("outflow")) - (pl.col("next_occupancy")) / pl.col("link_length"))
             ).pow(2).mean().alias("rmse")
        )
        average_error = rmse_data["rmse"].mean()
        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error
            print("Saved error for params: ", str_key, " with value: ", average_error)
            self.save_errors()
        
        
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

        plt.title("Density Error Heatmap")
        #plt.xlabel("Link Index")
        plt.ylabel("Time (s)")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(figure_path)
        plt.close()  
    
    def plot_point_queue_spatial_queue(self,
            data_file_name: str,
            hash_params: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None
    ):
        """
        Plotting the actual and predicted values for PointQueue and SpatialQueue models.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_params (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
            params (tuple, optional): Parameters to save with the error. Defaults to None.
        """
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_params}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        # Read geo
        cells_df = pl.read_csv(
            f"{self.cache_dir}/cells_{hash_geo}.csv"
        )
        cells_dict = cells_df.group_by("link_id").agg(
            pl.col("cell_id").count().alias("num_cells"),
            pl.col("cell_length_meters").sum().alias("link_length")
        ).to_dict(as_series=False)
        cells_dict = {cells_dict["link_id"][i]: {"num_cells": cells_dict["num_cells"][i], "link_length": cells_dict["link_length"][i]} for i in range(len(cells_dict["link_id"]))}

        with open(file_name, 'r') as f:
            json_data = json.load(f)
            
            # Flatten the nested structures if needed
            flattened_data = []
            for record in json_data:
                flat_record = {}
                for key, value in record.items():
                    if isinstance(value, (list, dict)):
                        # Convert complex types to string for visualization
                        if isinstance(value, list):
                            flat_record[key] = value
                        elif isinstance(value, dict):
                            v = dict(sorted(value.items(), key=lambda x: x[0]))
                            flat_record[key] = list(v.values())
                        
                    else:
                        flat_record[key] = value
                flattened_data.append(flat_record)
            data = pl.DataFrame(flattened_data, strict=False)
        data = data.filter((pl.col("link_id") != 5))
        link_lengths = self._get_link_lengths(hash_geo)
        if not link_lengths:
            raise ValueError(f"No link lengths found for geo hash: {hash_geo}")
        data = data.with_columns(
            pl.col("link_id").cast(pl.Int64).replace(link_lengths).alias("link_length"),
            pl.col("link_id").cast(pl.Int64)
        )

        
        

        all_rmse_data = []
        average_error = 0
        n = 0
        actual_min, actual_max = float("inf"), float("-inf")
        predicted_min, predicted_max = float("inf"), float("-inf")

        predicted_min, predicted_max = float("inf"), float("-inf")
        predicted_min_flow, predicted_max_flow = float("inf"), float("-inf")
        for row in tqdm(data.iter_rows(named=True), desc="Collecting error data for all links"):
            link_id = int(row["link_id"])
            trajectory_time = row["trajectory_time"]
            new_occupancy = max(0, row["new_occupancy"])
            next_occupancy = max(0, row["next_occupancy"])
            predicted_density = new_occupancy / cells_dict[link_id]["link_length"]
            actual_density = next_occupancy / cells_dict[link_id]["link_length"]
            squared_error = (actual_density - predicted_density)
            outflow = max(0, row["outflow"])
            sum_actual_outflow = max(0, row["actual_outflow"][0])
            sum_outflow = outflow
            flow_error = (sum_actual_outflow - sum_outflow)

            for i in range(cells_dict[link_id]["num_cells"]):
                
                all_rmse_data.append({
                    "link_id": link_id,
                    "trajectory_time": trajectory_time,
                    "squared_error": squared_error,
                    "actual_cell_density": actual_density,
                    "predicted_cell_density": predicted_density,
                    "cell_id": i + 1,
                    "link_cell_id": f"link {link_id} cell {i+1}",
                    "actual_flow": sum_actual_outflow,
                    "predicted_flow": sum_outflow,
                    "flow_error": flow_error
                })
                # print(squared_error)
                average_error += squared_error
                n += 1

            actual_min = min(actual_min, actual_density)
            actual_max = max(actual_max, actual_density)
            predicted_min = min(predicted_min, predicted_density)
            predicted_max = max(predicted_max, predicted_density)
            predicted_min_flow = min(predicted_min_flow, sum_actual_outflow, sum_outflow)
            predicted_max_flow = max(predicted_max_flow, sum_actual_outflow, sum_outflow)
            predicted_max_flow = max(predicted_max_flow, sum_actual_outflow, sum_outflow)
        if not all_rmse_data:
                print("No error data to plot.")
                return

        df = pd.DataFrame(all_rmse_data)
        sorted_cols = sorted(df["link_cell_id"].unique(), key=lambda x: (int(x.split()[1]), int(x.split()[3])))

        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        os.makedirs(figure_path, exist_ok=True)

        # --- Error Heatmap ---
        error_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="squared_error")
        error_data = error_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            error_data,
            cmap="Reds",
            vmin=df["squared_error"].min(),
            vmax=df["squared_error"].max(),
            cbar_kws={'label': r'Density Error $\frac{Veh}{m}$'}
        )
        plt.title(f"Error Heatmap for All Links ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "error_density.png")
        plt.close()

        # --- Actual Density Heatmap ---
        actual_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="actual_cell_density")
        actual_data = actual_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            actual_data,
            cmap="Reds",
            vmin=actual_min,
            vmax=actual_max,
            cbar_kws={'label': r'Actual Density $(Veh/m)$'}
        )
        plt.title(f"Actual Density Heatmap ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "actual_density.png")
        plt.close()

        # --- Predicted Density Heatmap ---
        predicted_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="predicted_cell_density")
        predicted_data = predicted_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            predicted_data,
            cmap="Reds",
            vmin=predicted_min,
            vmax=predicted_max,
            cbar_kws={'label': r'Predicted Density $(Veh/m)$'}
        )
        plt.title(f"Predicted Density Heatmap ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "predicted_density.png")
        plt.close()

        # --- Save error if params provided ---
        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error / n
            self.save_errors()

        # Flow
        flow_error_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="flow_error")

        plt.figure(figsize=(15, 8))
        sns.heatmap(
            flow_error_data,
            vmin=predicted_min_flow,
            vmax=predicted_max_flow,
            cmap="Reds",
            cbar_kws={'label': r'Flow Error $(Veh/s)$'}
        )
        plt.title(f"Flow Error Heatmap ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "flow_error.png")
        plt.close()

        # Actual Flow
        actual_flow_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="actual_flow")
        actual_flow_data = actual_flow_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            actual_flow_data,
            cmap="Reds",
            vmin=predicted_min_flow,
            vmax=predicted_max_flow,
            cbar_kws={'label': r'Actual Flow $(Veh/s)$'}
        )
        plt.title(f"Actual Flow Heatmap ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "actual_flow.png")
        plt.close()
        # Predicted Flow
        predicted_flow_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="predicted_flow")
        predicted_flow_data = predicted_flow_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            predicted_flow_data,
            cmap="Reds",
            vmin=predicted_min_flow,
            vmax=predicted_max_flow,
            cbar_kws={'label': r'Predicted Flow $(Veh/s)$'}
        )
        plt.title(f"Predicted Flow Heatmap ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "predicted_flow.png")
        plt.close()


    def plot_ctm(self,
                       data_file_name: str,
                       hash_params: str,
                       hash_geo: str,
                       traffic_model: str,
                       params: Optional[tuple] = None):
        """
        Plots the error in CTM.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_params (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
            params (tuple, optional): Parameters to save with the error. Defaults to None.
        """
        average_error = 0
        n = 0
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_params}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")

        try:
            # Try with increased schema length
            data = pl.read_json(file_name, infer_schema_length=1000)
        except Exception:
            # Manual fallback for complex nested JSON
            import json
            with open(file_name, 'r') as f:
                json_data = json.load(f)
            
            # Flatten the nested structures if needed
            flattened_data = []
            for record in json_data:
                flat_record = {}
                for key, value in record.items():
                    if isinstance(value, (list, dict)):
                        # Convert complex types to string for visualization
                        if isinstance(value, list):
                            flat_record[key] = value
                        elif isinstance(value, dict):
                            v = dict(sorted(value.items(), key=lambda x: x[0]))
                            flat_record[key] = list(v.values())
                        
                    else:
                        flat_record[key] = value
                flattened_data.append(flat_record)
            
            data = pl.DataFrame(flattened_data, strict=False)
        
                
        data = data.filter(pl.col("link_id") != 5) # Filter out link_id 5
        cell_length = self._get_cell_length(hash_geo)
        data = data.with_columns(
            pl.col("link_id")
            .cast(pl.Int64)
            .map_elements(lambda link_id: list(cell_length.get(link_id, {}).values()))
            .alias("cell_lengths")
        )
        
        data = data.with_columns(
            pl.struct("new_occupancy", "cell_lengths")
            .map_elements(lambda x: np.array(x["new_occupancy"]) / np.array(x["cell_lengths"]), return_dtype=pl.List(pl.Float64))
            .alias("new_densities")
        )
        data = data.with_columns(
            pl.struct("next_occupancy", "cell_lengths")
            .map_elements(lambda x: np.array(x["next_occupancy"]) / np.array(x["cell_lengths"]), return_dtype=pl.List(pl.Float64))
            .alias("next_densities")
        )
        data = data.with_columns(
            pl.struct(["new_densities", "next_densities"])
            .map_elements(lambda x: (np.array(x["new_densities"]) - np.array(x["next_densities"])) ** 2)
            .alias("squared_error")
        )

        error_info = data.select(
            ["link_id", "trajectory_time", "squared_error", "next_occupancy", "new_occupancy", "cell_lengths"]
        )

        # Prepare a list to hold all error data for the combined heatmap
        all_rmse_data = []

        # Calculate min and max for consistent colormap scaling
        all_errors_flat = []
        for row in error_info.iter_rows(named=True):
            all_errors_flat.extend(row["squared_error"])
        
        all_errors_flat = [e for e in all_errors_flat if e is not None] # Filter out None values

        actual_max = float("-inf")
        actual_min = float("inf")

        predicted_max = float("-inf")
        predicted_min = float("inf")
        all_rmse_data = []
        average_error = 0
        n = 0
        actual_min, actual_max = float("inf"), float("-inf")
        predicted_min, predicted_max = float("inf"), float("-inf")
        predicted_min_flow, predicted_max_flow = float("inf"), float("-inf")
        for row in tqdm(data.iter_rows(named=True), desc="Collecting error data for all links"):
            link_id = int(row["link_id"])
            trajectory_time = row["trajectory_time"]
            new_densities = row["new_densities"]
            next_density = row["next_densities"]
            inflow = row["inflow"]
            outflow = row["new_outflow"]
            actual_outflow = row["actual_outflow"]
            # print(inflow, outflow, actual_outflow)
            if isinstance(inflow, dict):
                actual_outflow = list(dict(list(sorted(actual_outflow.items(), key=lambda x: x[0]))).values())
            # print()
            # print()
            # print()
            # print()
            # print(actual_data)
            for i in range(len(next_density)):
                actual_density = next_density[i]
                predicted_density = max(new_densities[i], 0)
                squared_error = (actual_density - predicted_density) ** 2
                squared_flow_error = (actual_outflow[i] - outflow[i])
                all_rmse_data.append({
                    "link_id": link_id,
                    "trajectory_time": trajectory_time,
                    "squared_error": squared_error,
                    "actual_cell_density": actual_density,
                    "predicted_cell_density": predicted_density,
                    "cell_id": i + 1,
                    "link_cell_id": f"link {link_id} cell {i+1}",
                    "actual_flow": actual_outflow[i],
                    "predicted_flow": outflow[i],
                    "flow_error": squared_flow_error
                })
                # print(squared_error)
                average_error += squared_error
                n += 1

                actual_min = min(actual_min, actual_density)
                actual_max = max(actual_max, actual_density)
                predicted_min = min(predicted_min, predicted_density)
                predicted_max = max(predicted_max, predicted_density)
                predicted_min_flow = min([predicted_min_flow, actual_outflow[i], outflow[i]])
                predicted_max_flow = max(predicted_max_flow, actual_outflow[i], outflow[i])
        if not all_rmse_data:
                print("No error data to plot.")
                return

        df = pd.DataFrame(all_rmse_data)
        sorted_cols = sorted(df["link_cell_id"].unique(), key=lambda x: (int(x.split()[1]), int(x.split()[3])))

        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        os.makedirs(figure_path, exist_ok=True)

        # --- Error Heatmap ---
        error_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="squared_error")
        error_data = error_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        yticks = df.trajectory_time.unique().tolist()

        sns.heatmap(
            error_data,
            cmap="Reds",
            vmin=df["squared_error"].min(),
            vmax=df["squared_error"].max(),
            cbar_kws={'label': r'Density Error $\frac{Veh}{m}$'}
        )
        plt.title(f"Error Heatmap for All Links ({traffic_model})", fontsize=16)
        plt.xlabel("")
        plt.ylabel("Trajectory Time (s)", fontsize=14)
        tick_positions = np.arange(0, len(yticks), 50)
        tick_labels = [yticks[i] for i in tick_positions]
        plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "error_density.png")
        plt.close()

        # --- Actual Density Heatmap ---
        actual_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="actual_cell_density")
        actual_data = actual_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            actual_data,
            cmap="Reds",
            vmin=actual_min,
            vmax=actual_max,
            cbar_kws={'label': r'Actual Density $(Veh/m)$'}
        )
        plt.title(f"Actual Density Heatmap ({traffic_model})", fontsize=16)
        tick_positions = np.arange(0, len(yticks), 50)
        tick_labels = [yticks[i] for i in tick_positions]
        plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0)
        plt.xlabel("")
        plt.ylabel("Trajectory Time (s)", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "actual_density.png")
        plt.close()

        # --- Predicted Density Heatmap ---
        predicted_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="predicted_cell_density")
        predicted_data = predicted_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            predicted_data,
            cmap="Reds",
            vmin=predicted_min,
            vmax=predicted_max,
            cbar_kws={'label': r'Predicted Density $(Veh/m)$'}
        )
        plt.title(f"Predicted Density Heatmap ({traffic_model})")
        tick_positions = np.arange(0, len(yticks), 50)
        tick_labels = [yticks[i] for i in tick_positions]
        plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0)
        plt.xlabel("")
        plt.ylabel("Trajectory Time (s)", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(figure_path + "predicted_density.png")
        plt.close()
        
        # --- Save error if params provided ---
        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error / n
            self.save_errors()

        # Flow
        flow_error_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="flow_error")

        plt.figure(figsize=(15, 8))
        sns.heatmap(
            flow_error_data,
            vmin=predicted_min_flow,
            vmax=predicted_max_flow,
            cmap="Reds",
            cbar_kws={'label': 'Flow Error'}
        )
        plt.title(f"Flow Error Heatmap ({traffic_model})")
        tick_positions = np.arange(0, len(yticks), 50)
        tick_labels = [yticks[i] for i in tick_positions]
        plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0)
        plt.xlabel("")
        plt.ylabel("Trajectory Time (s)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figure_path + "flow_error.png")
        plt.close()
        

        # Actual Flow
        actual_flow_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="actual_flow")
        actual_flow_data = actual_flow_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            actual_flow_data,
            cmap="Reds",
            vmin=predicted_min_flow,
            vmax=predicted_max_flow,
            cbar_kws={'label': r'Actual Flow $(Veh/s)$'}
        )
        plt.title(f"Actual Flow Heatmap ({traffic_model})")
        tick_positions = np.arange(0, len(yticks), 50)
        tick_labels = [yticks[i] for i in tick_positions]
        plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0)
        plt.xlabel("")
        plt.ylabel("Trajectory Time (s)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figure_path + "actual_flow.png")
        plt.close()
        # Predicted Flow
        predicted_flow_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="predicted_flow")
        predicted_flow_data = predicted_flow_data[sorted_cols]
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            predicted_flow_data,
            cmap="Reds",
            vmin=predicted_min_flow,
            vmax=predicted_max_flow,
            cbar_kws={'label': r'Predicted Flow $(Veh/s)$'}
        )
        plt.title(f"Predicted Flow Heatmap ({traffic_model})")
        plt.xlabel("")
        plt.ylabel("Trajectory Time (s)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figure_path + "predicted_flow.png")
        plt.close()


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
                                                   hash_params: str,
                                                   hash_geo: str,
                                                   traffic_model: str):
        """
        Plotting the heatmap using seaborn.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_params (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
        """
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_params}.json"
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
        #plt.xlabel("Link ID")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.savefig(figure_path)
            
        
    def plot_ltm(self,
                       data_file_name,
                       hash_params: str,
                       hash_geo: str,
                       traffic_model: str,
                       params: Optional[tuple] = None):
        """
        Plotting the error in LTM.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_params (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
        """
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_params}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        data = pl.read_json(
            file_name
        )
        data = data.filter(
            pl.col("link_id") != 5
        )
        cell_length = self._get_cell_length(hash_geo)
        data = data.with_columns(
            pl.struct(["link_id", "cell_id"])
            .map_elements(lambda row: cell_length[int(row["link_id"])][int(row["cell_id"])], return_dtype=pl.Float64)
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
            ["link_id", "trajectory_time", "squared_error", "x", "new_occupancy", "next_occupancy", "cell_lengths"]
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
        # Step 1: Prepare a list to hold all error data
        all_rmse_data = []
        offset = 0.0

        for name, group in groups:
            link_id = int(name[0])  # type: ignore
            group = group.sort("trajectory_time")
            group = group.with_columns(
                (pl.col("x") + offset).alias("x")
            )
            max_x = float(group["x"].max()) # type: ignore
            offset += max_x  # update for next link

            group = group.to_pandas()
            for _, row in group.iterrows():
                all_rmse_data.append({
                    "trajectory_time": row["trajectory_time"],
                    "x": row["x"],
                    "squared_error": row["squared_error"],
                    "predicted_density": row["new_occupancy"] / row["cell_lengths"],
                    "actual_density": row["next_occupancy"] / row["cell_lengths"]

                })
        if not all_rmse_data:
            print("No error data to plot.")
            return

        # Step 2: Convert to DataFrame and pivot
        df_all = pd.DataFrame(all_rmse_data)
        heatmap_data = df_all.pivot(index="trajectory_time", columns="x", values="squared_error")
        heatmap_data = heatmap_data.sort_index(axis=1)

        # Step 3: Plot single heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            heatmap_data,
            cmap="Reds",
            vmin=min_errors, # type: ignore
            vmax=max_errors, # type: ignore
            annot=False,
            cbar_kws={'label': r'Density Error $\frac{Veh}{m}$'}
        )
        plt.title(f"Density Error Heatmap ({traffic_model})")
        #plt.xlabel("X (m, shifted)")
        plt.ylabel("Trajectory Time (s)", fontsize=14)
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')

        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + "error_density.png")
        plt.close()


        heatmap_data_actual_density = df_all.pivot(index="trajectory_time", columns="x", values="actual_density")
        heatmap_data_actual_density = heatmap_data_actual_density.sort_index(axis=1)
        _min = df_all["actual_density"].min()
        _max = df_all["actual_density"].max()
        # Step 3: Plot single heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            heatmap_data_actual_density,
            cmap="Reds",
            vmin=_min,
            vmax=_max,
            annot=False,
            cbar_kws={'label': 'Density Error'}
        )
        plt.title(f"Actual Density Heatmap ({traffic_model})")
        #plt.xlabel("X (m, shifted)")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')

        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + "actual_density.png")
        plt.close()

        heatmap_data_actual_density = df_all.pivot(index="trajectory_time", columns="x", values="predicted_density")
        heatmap_data_actual_density = heatmap_data_actual_density.sort_index(axis=1)
        _min = df_all["predicted_density"].min()
        _max = df_all["predicted_density"].max()
        # Step 3: Plot single heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            heatmap_data_actual_density,
            cmap="Reds",
            vmin=_min,
            vmax=_max,
            annot=False,
            cbar_kws={'label': 'Density Error'}
        )
        plt.title(f"Predicted Density Heatmap ({traffic_model})")
        #plt.xlabel("X (m, shifted)")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')

        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + "predicted_density.png")
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
            hash_params: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None
        ):
            # TODO: next_densities is actually next_occupancy. if you go about making this correct you should also
            # take care of pw.py in model folder.
            file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_params}.json"
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"File not found: {file_name}")

            data = pl.read_json(file_name)
            data = data.filter(pl.col("link_id") != 5)
            data = data.with_columns(
                pl.struct(["next_densities", "cell_lengths"])
                .map_elements(
                    lambda row: (np.array(row["next_densities"]) / np.array(row["cell_lengths"])).tolist(),
                    return_dtype=pl.List(pl.Float64)
                )
                .alias("next_densities")
            ) 

            all_rmse_data = []
            average_error = 0
            n = 0
            actual_min, actual_max = float("inf"), float("-inf")
            predicted_min, predicted_max = float("inf"), float("-inf")

            predicted_min, predicted_max = float("inf"), float("-inf")
            predicted_min_flow, predicted_max_flow = float("inf"), float("-inf")
            for row in tqdm(data.iter_rows(named=True), desc="Collecting error data for all links"):
                link_id = int(row["link_id"])
                trajectory_time = row["trajectory_time"]
                new_densities = row["new_densities"]
                next_density = row["next_densities"]
                inflow = row["inflow"]
                if isinstance(inflow, dict):
                    inflow = dict(sorted(inflow.items(), key=lambda x: x[0]))
                    inflow = list(inflow.values())
                outflow = row["outflow"]
                
                for i in range(len(next_density)):
                    actual_density = next_density[i]
                    predicted_density = max(new_densities[i], 0)
                    squared_error = (actual_density - predicted_density)
                    inflow_i = inflow[i]
                    outflow_i = outflow[i]
                    squared_flow_error = (inflow_i - outflow_i)

                    all_rmse_data.append({
                        "link_id": link_id,
                        "trajectory_time": trajectory_time,
                        "squared_error": squared_error,
                        "actual_cell_density": actual_density,
                        "predicted_cell_density": predicted_density,
                        "cell_id": i + 1,
                        "link_cell_id": f"link {link_id} cell {i+1}",
                        "actual_flow": inflow[i],
                        "predicted_flow": outflow[i],
                        "flow_error": squared_flow_error
                    })
                    # print(squared_error)
                    average_error += squared_error
                    n += 1

                    actual_min = min(actual_min, actual_density)
                    actual_max = max(actual_max, actual_density)
                    predicted_min = min(predicted_min, predicted_density)
                    predicted_max = max(predicted_max, predicted_density)
                    predicted_min_flow = min([predicted_min_flow, inflow_i, outflow_i])
                    predicted_max_flow = max(predicted_max_flow, inflow_i, outflow_i)
            if not all_rmse_data:
                    print("No error data to plot.")
                    return

            df = pd.DataFrame(all_rmse_data)
            sorted_cols = sorted(df["link_cell_id"].unique(), key=lambda x: (int(x.split()[1]), int(x.split()[3])))

            figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
            os.makedirs(figure_path, exist_ok=True)

            # --- Error Heatmap ---
            error_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="squared_error")
            error_data = error_data[sorted_cols]
            plt.figure(figsize=(15, 8))
            sns.heatmap(
                error_data,
                cmap="Reds",
                vmin=df["squared_error"].min(),
                vmax=df["squared_error"].max(),
                cbar_kws={'label': r'Density Error $\frac{Veh}{m}$'}
            )
            plt.title(f"Error Heatmap for All Links ({traffic_model})")
            #plt.xlabel("Link_ID and Cell_ID")
            plt.ylabel("Trajectory Time")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figure_path + "error_density.png")
            plt.close()

            # --- Actual Density Heatmap ---
            actual_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="actual_cell_density")
            actual_data = actual_data[sorted_cols]
            plt.figure(figsize=(15, 8))
            sns.heatmap(
                actual_data,
                cmap="Reds",
                vmin=actual_min,
                vmax=actual_max,
                cbar_kws={'label': r'Actual Density $(Veh/m)$'}
            )
            plt.title(f"Actual Density Heatmap ({traffic_model})")
            #plt.xlabel("Link_ID and Cell_ID")
            plt.ylabel("Trajectory Time")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figure_path + "actual_density.png")
            plt.close()

            # --- Predicted Density Heatmap ---
            predicted_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="predicted_cell_density")
            predicted_data = predicted_data[sorted_cols]
            plt.figure(figsize=(15, 8))
            sns.heatmap(
                predicted_data,
                cmap="Reds",
                vmin=predicted_min,
                vmax=predicted_max,
                cbar_kws={'label': r'Predicted Density $(Veh/m)$'}
            )
            plt.title(f"Predicted Density Heatmap ({traffic_model})")
            #plt.xlabel("Link_ID and Cell_ID")
            plt.ylabel("Trajectory Time")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figure_path + "predicted_density.png")
            plt.close()

            # --- Save error if params provided ---
            if params is not None:
                if traffic_model not in self.errors:
                    self.errors[traffic_model] = {}
                str_key = str(params)
                self.errors[traffic_model][str_key] = average_error / n
                self.save_errors()

            # Flow
            flow_error_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="flow_error")

            plt.figure(figsize=(15, 8))
            sns.heatmap(
                flow_error_data,
                vmin=predicted_min_flow,
                vmax=predicted_max_flow,
                cmap="Reds",
                cbar_kws={'label': 'Flow Error'}
            )
            plt.title(f"Flow Error Heatmap ({traffic_model})")
            #plt.xlabel("Link_ID and Cell_ID")
            plt.ylabel("Trajectory Time")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figure_path + "flow_error.png")
            plt.close()

            # Actual Flow
            actual_flow_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="actual_flow")
            actual_flow_data = actual_flow_data[sorted_cols]
            plt.figure(figsize=(15, 8))
            sns.heatmap(
                actual_flow_data,
                cmap="Reds",
                vmin=actual_min,
                vmax=actual_max,
                cbar_kws={'label': r'Actual Flow $(Veh/s)$'}
            )
            plt.title(f"Actual Flow Heatmap ({traffic_model})")
            #plt.xlabel("Link_ID and Cell_ID")
            plt.ylabel("Trajectory Time")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figure_path + "actual_flow.png")
            plt.close()
            # Predicted Flow
            predicted_flow_data = df.pivot(index="trajectory_time", columns="link_cell_id", values="predicted_flow")
            predicted_flow_data = predicted_flow_data[sorted_cols]
            plt.figure(figsize=(15, 8))
            sns.heatmap(
                predicted_flow_data,
                cmap="Reds",
                vmin=predicted_min_flow,
                vmax=predicted_max_flow,
                cbar_kws={'label': r'Predicted Flow $(Veh/s)$'}
            )
            plt.title(f"Predicted Flow Heatmap ({traffic_model})")
            #plt.xlabel("Link_ID and Cell_ID")
            plt.ylabel("Trajectory Time")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figure_path + "predicted_flow.png")
            plt.close()


        
        

    def plot(self,
             data_file_name: str,
            hash_params: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None):
        """
        Plotting the data
        """
        if traffic_model == "LTM":
            # self.plot_error_ltm(
            #     data_file_name=data_file_name,
            #     hash_params=hash_params,
            #     hash_geo=hash_geo,
            #     traffic_model=traffic_model,
            #     params=params
            # )
            pass
        elif traffic_model == "CTM":
            self.plot_ctm(
                data_file_name=data_file_name,
                hash_params=hash_params,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
        elif traffic_model == "PointQueue" or traffic_model == "SpatialQueue":
            self.plot_point_queue_spatial_queue(
                data_file_name=data_file_name,
                hash_params=hash_params,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
            # self.plot_actual_predicted_point_queue_spatial_queue(
            #     data_file_name=data_file_name,
            #     hash_params=hash_params,
            #     hash_geo=hash_geo,
            #     traffic_model=traffic_model,
            # )
        elif traffic_model == "PW":
            self.plot_error_pw(
                data_file_name=data_file_name,
                hash_params=hash_params,
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

    def plot_trajectory(
            self,
            data_file_name: str,
            hash_geo: str,
            min_time: Optional[float] = None,
            max_time: Optional[float] = None
    ):
        """
        Plotting the trajectory of the vehicles.
        """
        file_name = f"{self.cache_dir}/{data_file_name}_vehicle_on_corridor_{hash_geo}.csv"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        df = pd.read_csv(file_name)
        geo_loader = self._get_geo_loader(hash_geo)
        links = geo_loader.get_links()
        
        df = df[df["trajectory_time"] >= min_time] if min_time is not None else df
        df = df[df["trajectory_time"] <= max_time] if max_time is not None else df
        df = df[df["distance_from_cell"] <= 100]
        print(f"Plotting trajectory for {len(df)} vehicles.")
        
        print("Sorting the dataframe by trajectory time and loc.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        distance_passed = 0.0
        print("Plotting the trajectory of the vehicles.")
        links = dict(sorted(links.items(), key=lambda x: x[0]))  # sort links by id
        add_link_distance = {}
        for link_id, link in links.items():
            add_link_distance[link_id] = distance_passed
            distance_passed += link.get_length().value
            ax.axhline(y=distance_passed, color='r', linestyle='--', linewidth=2)
        print("Distance passed:", distance_passed)
        ax.set_xlabel("Trajectory Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Trajectory of Vehicles")
        ax.set_xlim(df["trajectory_time"].min(), df["trajectory_time"].max())
        ax.set_ylim(0, distance_passed + 100)
        df["loc"] = df.apply(
            lambda row: GEOD.inv(
                row["lon"], row["lat"],
                links[row["link_id"]].get_from().x,
                links[row["link_id"]].get_from().y
            )[2] + add_link_distance[row["link_id"]],  # only get the distance
            axis=1
        )
        df = df.sort_values(by=["trajectory_time", "loc"])
        ax.plot(df["trajectory_time"], df["loc"], marker='o', linestyle='-', markersize=0.1, linewidth=0.01, label="Vehicle Trajectory")    
        ax.legend()
        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + "trajectory_plot.png")
        plt.close()

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
    def plot_all(self):
        """
        Plot all the existing models results.
        """
        for folder in os.listdir(self.cache_dir):
            # If the first characters of the folder name are capital letters, it is a traffic model
            if folder[0].isupper():
                traffic_model = folder
                for file_name in os.listdir(f"{self.cache_dir}/{traffic_model}"):
                    if file_name.endswith(".json"):
                        data_file_name = "_".join(file_name.split("_")[:4])
                        hash_geo = file_name.split("_")[4].split(".")[0]
                        hash_params = file_name.split("_")[5].split(".")[0]
                        with open(f"{self.cache_dir}/params/{hash_params}.json", "r") as f:
                            params = json.load(f)
                            free_flow_speed = float(params["free_flow_speed"].split(" ")[0])
                            wave_speed = float(params["wave_speed"].split(" ")[0])
                            dt = float(params["dt"].split(" ")[0])
                            jam_density_link = float(params["jam_density_link"].split(" ")[0])
                            q_max = float(params["q_max"].split(" ")[0])

                        self.plot(
                            data_file_name=data_file_name,
                            hash_params=hash_params,
                            hash_geo=hash_geo,
                            traffic_model=traffic_model,
                            params=(free_flow_speed, wave_speed, dt, jam_density_link, q_max)
                        )
    def plot_fundamental_diagram(self):
        """
        Plot the fundamental diagram.
        """
        folder = f"{self.cache_dir}/params"
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        plt.figure(figsize=(12, 7), dpi=300)  # higher resolution and larger figure
        color_map = cm.get_cmap('tab20')  # or 'Set1', 'hsv', etc.
        param_files = [f for f in os.listdir(folder) if f.endswith(".json")]
        np.random.seed(42)  # for reproducibility
        param_files = np.random.choice(param_files, size=8, replace=False)
        colors = color_map(np.linspace(0, 1, len(param_files)))

        for i, file_name in enumerate(param_files):
            with open(f"{folder}/{file_name}", "r") as f:
                params = json.load(f)
                free_flow_speed = float(params["free_flow_speed"].split(" ")[0]) * Units.KM_PER_HR
                wave_speed = float(params["wave_speed"].split(" ")[0]) * Units.KM_PER_HR
                jam_density_link = float(params["jam_density_link"].split(" ")[0]) * Units.PER_KM
                q_max = float(params["q_max"].split(" ")[0]) * Units.PER_HR

                x_intercept_ffs = (q_max / free_flow_speed).to(Units.PER_KM).value
                x_intercept_ws = (((jam_density_link * wave_speed) - q_max) / wave_speed).to(Units.PER_KM).value
                last_x = jam_density_link.to(Units.PER_KM).value
                y_intercept = q_max.to(Units.PER_HR).value
                label = rf"FFS: {free_flow_speed.to(Units.KM_PER_HR).value:.1f} $\frac{{\mathrm{{km}}}}{{\mathrm{{hr}}}}$, " \
                rf"WS: {wave_speed.to(Units.KM_PER_HR).value:.1f} $\frac{{\mathrm{{km}}}}{{\mathrm{{hr}}}}$, " \
                rf"$K_j$: {jam_density_link.to(Units.PER_KM).value:.1f} $\frac{{\mathrm{{veh}}}}{{\mathrm{{km}}}}$, " \
                rf"$Q_{{\max}}$: {q_max.to(Units.PER_HR).value:.1f} $\frac{{\mathrm{{veh}}}}{{\mathrm{{hr}}}}$"
                # Plot the free-flow segment with the label
                plt.plot([0, x_intercept_ffs], [0, y_intercept], color=colors[i], linewidth=1.5, label=label)

                # Plot the congested side
                if x_intercept_ws < x_intercept_ffs:
                    # Triangle FD
                    plt.plot([x_intercept_ffs, last_x], [y_intercept, 0], color=colors[i], linewidth=1.5)
                else:
                    # Trapezoidal FD
                    plt.plot([x_intercept_ffs, x_intercept_ws], [y_intercept, y_intercept], color=colors[i], linewidth=1.5)
                    plt.plot([x_intercept_ws, last_x], [y_intercept, 0], color=colors[i], linewidth=1.5)
        
        # === Axis labels with LaTeX formatting ===
        plt.xlabel(r"$K$ (jam density) $\left(\frac{\mathrm{veh}}{\mathrm{km}}\right)$", fontsize=14)
        plt.ylabel(r"$Q_{\max}$ $\left(\frac{\mathrm{veh}}{\mathrm{hr}}\right)$", fontsize=14)
        plt.title("Fundamental Diagrams", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.legend(loc='upper left', fontsize=10)

        # Optional: save in higher-quality formats
        output_path = f"{self.cache_dir}/results/fundamental_diagram"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path + ".png", dpi=300)
        plt.savefig(output_path + ".pdf", dpi=300)
        plt.close()



if __name__ == "__main__":
    
    data_file_name = "d1_20181029_0800_0830"
    params_hash = "0a043bb9c51bc73349c36a052764479b"
    geo_hash = "682a48de"
    traffic_model_name = "PW"
    plotter = Plotter(cache_dir=".cache_dt5s")
    # plotter.plot_all()
    plotter.plot_fundamental_diagram()
    # plotter.animation(f".cache/{data_file_name}_fully_process_vehicles_{geo_hash}.csv")
    # print("Heatmap generated and saved successfully.")
    
    # plotter.plot(
    #     data_file_name=data_file_name,
    #     hash_params=params_hash,
    #     hash_geo=geo_hash,
    #     traffic_model=traffic_model_name
    # )
# row