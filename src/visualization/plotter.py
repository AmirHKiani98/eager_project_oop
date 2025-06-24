 
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
        
        data = pl.read_json(
            file_name
        )
        data = data.filter(
            pl.col("link_id") != 5
        )
        data = data.with_columns(
            pl.col("link_id").cast(pl.Int64).replace(self.link_lengths).alias("link_length")
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
        plt.xlabel("Link Index")
        plt.ylabel("Time (s)")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(figure_path)
        plt.close()  
    
    def plot_point_queue_spatial_queue(self,
            data_file_name: str,
            hash_parmas: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None
    ):
        """
        Plotting the actual and predicted values for PointQueue and SpatialQueue models.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_parmas (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
            params (tuple, optional): Parameters to save with the error. Defaults to None.
        """
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_parmas}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        
        data = pl.read_json(file_name)
        data = data.filter((pl.col("link_id") != 5))
        data = data.with_columns(
            pl.col("link_id").cast(pl.Int64).replace(self.link_lengths).alias("link_length"),
            pl.col("link_id").cast(pl.Int64)
        )

        actual_figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/actual_density.png"
        predicted_figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/predicted_density.png"
        density_error = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/error_density.png"
        
        if not os.path.exists(os.path.dirname(actual_figure_path)):
            os.makedirs(os.path.dirname(actual_figure_path))
        data = data.with_columns([
            ((pl.col("next_occupancy") - pl.col("new_occupancy")).pow(2).pow(0.5)/pl.col("link_length")).alias("error_density"),
            (pl.col("next_occupancy") / pl.col("link_length")).alias("actual_density"),
            (pl.col("new_occupancy") / pl.col("link_length")).alias("predicted_density")
        ])

        data = data.sort("trajectory_time")
        data_pd = data.to_pandas()
        heatmap_data_actual = data_pd.pivot(index="trajectory_time", columns="link_id", values="actual_density")
        _min = data_pd["actual_density"].min()
        _max = data_pd["actual_density"].max()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data_actual,
            cmap="Reds",
            vmin=_min,
            vmax=_max,
            annot=False,
            cbar_kws={'label': 'Actual'}
        )

        plt.title("Actual Density Heatmap")
        plt.xlabel("Link Index")
        plt.ylabel("Time (s)")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(actual_figure_path)
        plt.close()
        
        
        heatmap_data_predicted = data_pd.pivot(index="trajectory_time", columns="link_id", values="predicted_density")
        _min = data_pd["predicted_density"].min()
        _max = data_pd["predicted_density"].max()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data_predicted,
            cmap="Reds",
            vmin=_min,
            vmax=_max,
            annot=False,
            cbar_kws={'label': 'Predicted'}
        )

        plt.title("Predicted Density Heatmap")
        plt.xlabel("Link Index")
        plt.ylabel("Time (s)")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(predicted_figure_path)
        plt.close()

        heatmap_data_density_error = data_pd.pivot(index="trajectory_time", columns="link_id", values="error_density")
        _min = data_pd["error_density"].min()
        _max = data_pd["error_density"].max()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data_density_error,
            cmap="Reds",
            vmin=_min,
            vmax=_max,
            annot=False,
            cbar_kws={'label': 'Error Density'}
        )
        average_error = data["error_density"].mean()
        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error
            print("Saved error for params: ", str_key, " with value: ", average_error)
            self.save_errors()
        
        plt.title("Density Error Heatmap")
        plt.xlabel("Link Index")
        plt.ylabel("Time (s)")
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(density_error)
        plt.close()



    def truncate_and_square_error_ctm(self, row):
        new_occ = np.array(row["new_occupancy"])
        next_occ = np.array(row["next_occupancy"])
        lengths = np.array(row["cell_lengths"])

        min_len = min(len(new_occ), len(next_occ), len(lengths))
        return ((new_occ[:min_len] - next_occ[:min_len]) / lengths[:min_len]) ** 2

    def plot_ctm(self,
                       data_file_name: str,
                       hash_parmas: str,
                       hash_geo: str,
                       traffic_model: str,
                       params: Optional[tuple] = None):
        """
        Plots the error in CTM.
        Args:
            data_file_name (str): The name of the file to plot.
            hash_parmas (str): The hash of the parameters.
            hash_geo (str): The hash of the geo.
            traffic_model (str): The name of the traffic model.
            params (tuple, optional): Parameters to save with the error. Defaults to None.
        """
        average_error = 0
        n = 0
        file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_parmas}.json"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")

        data = pl.read_json(file_name)
        data = data.filter(pl.col("link_id") != 5) # Filter out link_id 5

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
            ["link_id", "trajectory_time", "squared_error", "next_occupancy", "new_occupancy", "cell_lengths"]
        )

        # Prepare a list to hold all error data for the combined heatmap
        all_rmse_data = []

        # Calculate min and max for consistent colormap scaling
        all_errors_flat = []
        for row in error_info.iter_rows(named=True):
            all_errors_flat.extend(row["squared_error"])
        
        all_errors_flat = [e for e in all_errors_flat if e is not None] # Filter out None values
        rmse_data_min = min(all_errors_flat) if all_errors_flat else 0
        rmse_data_max = max(all_errors_flat) if all_errors_flat else 1


        groups = error_info.group_by(["link_id"])
        actual_max = float("-inf")
        actual_min = float("inf")

        predicted_max = float("-inf")
        predicted_min = float("inf")
        for name, group in groups:
            link_id = int(name[0]) # type: ignore
            group = group.sort("trajectory_time")
            for row in tqdm(group.iter_rows(named=True), desc=f"Collecting data for link {link_id}", total=len(group)):
                trajectory_time = row["trajectory_time"]
                squared_error = row["squared_error"]
                next_occupancy = row["next_occupancy"]
                new_occupancy = row["new_occupancy"]
                cell_lengths = row["cell_lengths"]
                for i in range(len(squared_error)):
                    actual_cell_density = next_occupancy[i] / cell_lengths[i]
                    predicted_cell_density = new_occupancy[i] / cell_lengths[i]
                    all_rmse_data.append({
                        "link_id": link_id,
                        "trajectory_time": trajectory_time,
                        "squared_error": squared_error[i],
                        "actual_cell_density": actual_cell_density,
                        "predicted_cell_density": predicted_cell_density,
                        "cell_id": i + 1
                    })
                    if actual_cell_density > actual_max:
                        actual_max = actual_cell_density
                    
                    if actual_cell_density < actual_min:
                        actual_max = actual_cell_density

                    if predicted_cell_density > predicted_max:
                        predicted_max = predicted_cell_density

                    if predicted_cell_density < predicted_min:
                        predicted_min = predicted_cell_density

                    average_error += squared_error[i]
                    n += 1

        if not all_rmse_data:
            print("No error data to plot.")
            return

        combined_rmse_df = pd.DataFrame(all_rmse_data)

        # Create a unique identifier for each cell within its link
        combined_rmse_df['link_cell_id'] = "link " + combined_rmse_df['link_id'].astype(str) + ' cell ' + combined_rmse_df['cell_id'].astype(str)

        # Pivot the data for the combined heatmap
        # We need to decide how to represent multiple links on a single heatmap.
        # One common way is to stack them vertically or horizontally.
        # For simplicity, let's create a "synthetic" cell ID that combines link and cell.
        combined_heatmap_data = combined_rmse_df.pivot_table(
            index="trajectory_time",
            columns="link_cell_id",
            values="squared_error"
        )
        
        combined_heatmap_data = combined_heatmap_data.reindex(sorted(combined_heatmap_data.columns, key=lambda x: (int(x.split(' ')[1]), int(x.split(' ')[3]))), axis=1)


        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        plt.figure(figsize=(15, 8)) 
        sns.heatmap(
            combined_heatmap_data,
            cmap="Reds",
            vmin=rmse_data_min,
            vmax=rmse_data_max,
            annot=False,
            cbar_kws={'label': 'Squared Error'},
            
        )
        plt.title(f"Heatmap of Squared Error for All Links ({traffic_model})")
        plt.xlabel("Link_ID and Cell_ID")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(figure_path + "error_density.png")
        plt.close()


        combined_heatmap_data_actual_density = combined_rmse_df.pivot_table(
            index="trajectory_time",
            columns="link_cell_id",
            values="actual_cell_density"
        )
        
        combined_heatmap_data_actual_density = combined_heatmap_data_actual_density.reindex(sorted(combined_heatmap_data_actual_density.columns, key=lambda x: (int(x.split(' ')[1]), int(x.split(' ')[3]))), axis=1)


        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        plt.figure(figsize=(15, 8))
        
        sns.heatmap(
            combined_heatmap_data_actual_density,
            cmap="Reds",
            vmin=actual_min,
            vmax=actual_max,
            annot=False,
            cbar_kws={'label': 'Actual Density'},
            
        )
        plt.title(f"Density Heatmap for All Links ({traffic_model})")
        plt.xlabel("Link_ID and Cell_ID")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(figure_path + "actual_density.png")
        plt.close()


        combined_heatmap_data_predicted_density = combined_rmse_df.pivot_table(
            index="trajectory_time",
            columns="link_cell_id",
            values="predicted_cell_density"
        )
        
        combined_heatmap_data_predicted_density = combined_heatmap_data_predicted_density.reindex(sorted(combined_heatmap_data_predicted_density.columns, key=lambda x: (int(x.split(' ')[1]), int(x.split(' ')[3]))), axis=1)


        figure_path = f"{self.cache_dir}/results/{self.get_base_name_without_extension(file_name)}/{traffic_model}/"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        plt.figure(figsize=(15, 8)) 
        sns.heatmap(
            combined_heatmap_data_predicted_density,
            cmap="Reds",
            vmin=predicted_min,
            vmax=predicted_max,
            annot=False,
            cbar_kws={'label': 'Actual Density'},
            
        )
        plt.title(f"Density Heatmap for All Links ({traffic_model})")
        plt.xlabel("Link_ID and Cell_ID")
        plt.ylabel("Trajectory Time")
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(figure_path + "predicted_density.png")
        plt.close()

        

        if params is not None:
            if traffic_model not in self.errors:
                self.errors[traffic_model] = {}
            
            str_key = str(params)
            self.errors[traffic_model][str_key] = average_error/n
            self.save_errors()


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
            
        
    def plot_ltm(self,
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
        data = data.filter(
            pl.col("link_id") != 5
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
            cbar_kws={'label': 'Squared Error'}
        )
        plt.title(f"Density Error Heatmap ({traffic_model})")
        plt.xlabel("X (m, shifted)")
        plt.ylabel("Trajectory Time")
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
            cbar_kws={'label': 'Squared Error'}
        )
        plt.title(f"Actual Density Heatmap ({traffic_model})")
        plt.xlabel("X (m, shifted)")
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
            cbar_kws={'label': 'Squared Error'}
        )
        plt.title(f"Predicted Density Heatmap ({traffic_model})")
        plt.xlabel("X (m, shifted)")
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
            hash_parmas: str,
            hash_geo: str,
            traffic_model: str,
            params: Optional[tuple] = None
        ):
            file_name = f"{self.cache_dir}/{traffic_model}/{data_file_name}_{hash_geo}_{hash_parmas}.json"
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"File not found: {file_name}")

            data = pl.read_json(file_name)
            data = data.filter(pl.col("link_id") != 5)

            data = data.with_columns(
                pl.struct(["new_densities", "next_densities"])
                .map_elements(
                    lambda row: ((np.array(row["new_densities"]) - np.array(row["next_densities"])) / len(row["new_densities"])) ** 2,
                    return_dtype=pl.List(pl.Float64)
                )
                .alias("squared_error")
            )

            all_rmse_data = []
            average_error = 0
            n = 0
            actual_min, actual_max = float("inf"), float("-inf")
            predicted_min, predicted_max = float("inf"), float("-inf")

            for row in tqdm(data.iter_rows(named=True), desc="Collecting error data for all links"):
                link_id = int(row["link_id"])
                trajectory_time = row["trajectory_time"]
                squared_error = row["squared_error"]
                new_densities = row["new_densities"]
                next_densities = row["next_densities"]
                cell_lengths = row["cell_lengths"]

                for i in range(len(squared_error)):
                    actual_density = next_densities[i]
                    predicted_density = new_densities[i]
                    all_rmse_data.append({
                        "link_id": link_id,
                        "trajectory_time": trajectory_time,
                        "squared_error": squared_error[i],
                        "actual_cell_density": actual_density,
                        "predicted_cell_density": predicted_density,
                        "cell_id": i + 1,
                        "link_cell_id": f"link {link_id} cell {i+1}"
                    })
                    average_error += squared_error[i]
                    n += 1

                    actual_min = min(actual_min, actual_density)
                    actual_max = max(actual_max, actual_density)
                    predicted_min = min(predicted_min, predicted_density)
                    predicted_max = max(predicted_max, predicted_density)

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
                cbar_kws={'label': 'Squared Error'}
            )
            plt.title(f"Error Heatmap for All Links ({traffic_model})")
            plt.xlabel("Link_ID and Cell_ID")
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
                cbar_kws={'label': 'Actual Density'}
            )
            plt.title(f"Actual Density Heatmap ({traffic_model})")
            plt.xlabel("Link_ID and Cell_ID")
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
                cbar_kws={'label': 'Predicted Density'}
            )
            plt.title(f"Predicted Density Heatmap ({traffic_model})")
            plt.xlabel("Link_ID and Cell_ID")
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
            self.plot_ctm(
                data_file_name=data_file_name,
                hash_parmas=hash_parmas,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
        elif traffic_model == "PointQueue" or traffic_model == "SpatialQueue":
            self.plot_point_queue_spatial_queue(
                data_file_name=data_file_name,
                hash_parmas=hash_parmas,
                hash_geo=hash_geo,
                traffic_model=traffic_model,
                params=params
            )
            # self.plot_actual_predicted_point_queue_spatial_queue(
            #     data_file_name=data_file_name,
            #     hash_parmas=hash_parmas,
            #     hash_geo=hash_geo,
            #     traffic_model=traffic_model,
            # )
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
    params_hash = "00b87522d0046c6d7054be1242574dad"
    geo_hash = "682a48de"
    traffic_model_name = "PW"
    plotter = Plotter(cache_dir=".cache", geo_loader=model_geo_loader)
    # plotter.animation(f".cache/{data_file_name}_fully_process_vehicles_{geo_hash}.csv")
    # print("Heatmap generated and saved successfully.")
    plotter.plot(
        data_file_name=data_file_name,
        hash_parmas=params_hash,
        hash_geo=geo_hash,
        traffic_model=traffic_model_name
    )
