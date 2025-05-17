"""
traffic_model.py
This module defines the `TrafficModel` abstract base class, which serves as a 
blueprint for implementing traffic simulation models. It provides a structured 
interface for setting parameters, retrieving cell lengths, and predicting 
traffic flow.

Classes:
    TrafficModel: An abstract base class for traffic simulation models.

Usage:
    This module is intended to be extended by specific implementations of 
    traffic models. Subclasses must implement the `predict` method to define 
    the logic for traffic flow prediction.
Dependencies:
    - abc.abstractmethod: Used to define abstract methods.
    - src.model.params.Parameters: Represents configuration and settings for 
      the traffic model.
    - src.preprocessing.geo_loader.GeoLoader: Provides geographical data 
      related to the traffic model.
"""
import math
import os
import json
import logging
import itertools
from abc import abstractmethod
from multiprocessing import Pool, cpu_count
from more_itertools import chunked
from rich.logging import RichHandler
import numpy as np
from tqdm import tqdm
from src.preprocessing.data_loader import DataLoader
from src.common_utility.units import Units

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
class TrafficModel:
    """
    TrafficModel is an abstract base class that represents a traffic simulation model. 
    It provides a structure for defining traffic-related operations, such as setting parameters, 
    retrieving cell lengths, and predicting traffic flow. This class is designed to be extended 
    by specific implementations of traffic models.
    Attributes:
        geo_loader (GeoLoader): An instance of GeoLoader used to retrieve geographical data 
            related to the traffic model, such as cell lengths.
        params (Parameters): An instance of Parameters containing configuration and settings 
            for the traffic model.
    Methods:
        set_params(params):
            Updates the parameters of the traffic model with a new Parameters object.
        get_cell_length(cell_id, link_id):
            Retrieves the length of a specific cell within a link using the GeoLoader instance.
        predict(**args):
            Abstract method that must be implemented by subclasses to predict traffic flow 
            based on the model's logic and input arguments.
    """
    def __init__(self, dl: DataLoader, fp_location: str, fp_date: str, fp_time: str):
        """
        Initialize the TrafficModel with a GeoLoader instance, Parameters object,
        and a DataLoader instance.
        Args:
            geo_loader (GeoLoader): An instance of GeoLoader for geographical data.
            params (Parameters): An instance of Parameters for model configuration.
            dl (DataLoader): An instance of DataLoader for loading data.
            fp_location (str): The location of the file.
            fp_date (str): The date of the file.
            fp_time (str): The time of the file.
        """
        self.dl = dl
        self.fp_location = fp_location
        self.fp_date = fp_date
        self.fp_time = fp_time


    def get_cell_length(self, cell_id, link_id):
        """
        Get the length of a specific cell.

        Args:
            cell_id (int): The ID of the cell.
            link_id (int): The ID of the link.

        Returns:
            float: Length of the specified cell.
        """
        return self.dl.geo_loader.get_cell_length(cell_id, link_id)


    def compute_outflow(
        self,
        free_flow_speed, dt, jam_density, wave_speed,
        max_flow, density_current, density_next=None
    ):
        """
        Computes the outflow of traffic based on the given parameters.

        Parameters:
            free_flow_speed (float): The speed of traffic under free-flow 
            conditions (e.g., vehicles per unit time).
            dt (float): The time step duration.
            jam_density (float): The maximum vehicle density (e.g., vehicles 
            per unit length) at which traffic is completely jammed.
            wave_speed (float): The speed at which traffic congestion 
            propagates backward.
            density_current (float): The current traffic density.
            density_next (float, optional): The traffic density at the next 
            location. Defaults to None.

        Returns:
            float: The computed outflow, which is the minimum of the maximum 
            flow, demand, and supply.
        """
        demand = free_flow_speed * density_current * dt
        if density_next is None:
            supply = math.inf
        else:
            supply = wave_speed * (jam_density - density_next) * dt
        return min(max_flow, demand, supply)


    def run_with_multiprocessing(self, num_processes=None, batch_size=None):
        """
        Run the `run()` method in parallel using batching and multiprocessing.

        Args:
            num_processes (int, optional): Number of worker processes. Defaults 
            to cpu_count().
            batch_size (int, optional): Number of tasks to process per batch. 
            If None, process all at once.

        Returns:
            list: Aggregated results from all batches.
        """
        self.dl.prepare(self.__class__.__name__, self.fp_location, self.fp_date, self.fp_time)
        args_list = self.dl.tasks
        if not args_list:
            raise ValueError("No tasks to process. Please provide a list of tasks.")
        if num_processes is None:
            num_processes = cpu_count()
        if batch_size is None:
            batch_size = int(len(args_list)/2)
        run_file_path = self.get_run_file_path()
        parent_dir = os.path.dirname(run_file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        if os.path.exists(run_file_path):
            # logger.debug(f"Run file already exists at {run_file_path}.")
            return []

        all_results = []
        with Pool(processes=num_processes) as pool:
            for batch in tqdm(
                chunked(args_list, batch_size),
                total=math.ceil(len(args_list) / batch_size),
                desc="Processing traffic model"
            ):
                results = pool.map(
                    type(self).run,
                    batch
                )
                all_results.extend(results)
        with open(run_file_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        return all_results

    @abstractmethod
    def predict(self, **args):
        """
        Abstract method to predict traffic flow.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    @abstractmethod
    def run(args):
        """
        Abstract method to run the traffic model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def compute_flow(self, **args):
        """
        Abstract method to compute flow.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def run_calibration(self, num_processes=None, batch_size=None):
        """
        This function runs the calibration process for the traffic model.
        It uses multiprocessing to speed up the process by dividing the tasks
        into batches and processing them in parallel.
        Args:
            num_processes (int, optional): Number of worker processes. Defaults 
            to cpu_count().
            batch_size (int, optional): Number of tasks to process per batch. 
            If None, process all at once.
        Returns:
            void
        """
        free_flow_speeds = np.linspace(30, 50, 5)
        jam_densities = np.linspace(120, 160, 5)
        wave_speeds = np.linspace(10, 20, 5)
        q_max = np.linspace(1000, 4000, 5)
        combinations = list(itertools.product(
            free_flow_speeds,
            jam_densities,
            wave_speeds,
            q_max
        ))
        combinations_array = np.array(combinations)
        for params in combinations_array:
            logger.info(
                "Running calibration with params: free_flow_speed: %s, jam_density: %s, "
                "wave_speed: %s, q_max: %s",
                params[0] * Units.KM_PER_HR,
                params[1] * Units.PER_KM,
                params[2] * Units.KM_PER_HR,
                params[3] * Units.PER_HR
            )
            self.dl.params.set_initialized(False)
            self.dl.params.free_flow_speed = params[0] * Units.KM_PER_HR
            self.dl.params.jam_density_link = params[1] * Units.PER_KM
            self.dl.params.wave_speed = params[2] * Units.KM_PER_HR
            self.dl.params.q_max = params[3] * Units.PER_HR
            self.dl.params.set_initialized(True)
            self.dl.params.save_metadata()
            
            
            self.run_with_multiprocessing(num_processes, batch_size)

    def get_run_file_path(self):
        """
        Get the file path of the run.

        Returns:
            str: The file path of the run.
        """
        return (
            (
                self.dl.params.cache_dir + "/"
                + self.__class__.__name__
                + "/"
                + self.dl.current_file_running["location"]
                + "_"
                + self.dl.current_file_running["date"]
                + "_"
                + self.dl.current_file_running["time"]
                + "_"
                + self.dl.geo_loader.get_hash_str()
                + "_"
                + self.dl.params.get_hash_str()
                + ".json"
            )
        )
