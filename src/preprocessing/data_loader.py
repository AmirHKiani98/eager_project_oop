"""
This module provides the `DataLoader` class, which is designed to handle the downloading, 
caching, and validation of traffic data files from a specified remote server. 
The `DataLoader` class includes methods for:
- Validating input parameters such as location, date, and time.
- Constructing download URLs and payloads.
- Downloading files and storing them in a local cache directory.
- Checking the existence and integrity of cached files.
The module is intended for use in scenarios where traffic data needs to be 
retrieved programmatically and efficiently managed in a local environment.
Example:
    To use the `DataLoader` class, initialize it with the desired parameters:
"""
import os
import json
from copy import deepcopy
from math import atan2, degrees
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import logging
from typing import Optional
import chardet
from sklearn.linear_model import LinearRegression
from more_itertools import chunked
from shapely.geometry import Point as POINT
import requests
from tqdm import tqdm
import numpy as np
import polars as pl
from rich.logging import RichHandler
from src.preprocessing.geo_loader import GeoLoader
from src.model.params import Parameters
from src.preprocessing.utility import fill_missing_timestamps
from src.common_utility.units import Units
from src.common_utility.utility import convert_keys_to_float
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

class DataLoader:
    """
    DataLoader is a class designed to handle the downloading and caching of traffic data files 
    from a specified remote server. It validates input parameters, constructs appropriate 
    download URLs, and ensures that files are downloaded and stored in a local cache directory. 
    The class also provides methods to check the existence and integrity of cached files.
    Attributes:
        base_url (str): The base URL for downloading files.
        fp_location (list): List of location identifiers.
        fp_date (list): List of dates in the format 'yyyymmdd'.
        fp_time (list): List of time ranges in the format 'hhmm_hhmm'.
        cache_dir (str): Directory where downloaded files are cached.
        files_list (dict): Dictionary to store information about downloaded files.
    """
    def __init__(
        self,
        fp_location: str | list,
        fp_date: str | list,
        fp_time: str | list,
        geo_loader: GeoLoader,
        params: Parameters,
        line_threshold=20,
        time_interval=0.04,
        traffic_light_speed_threshold=0.5,
        test_row_numbers=10
    ):
        """
        Initializes the DataLoader with the specified parameters.

        Args:
            fp_location (str | list): Specifies the location identifier(s).
            fp_date (str | list): Specifies the date(s) in the format 'yyyymmdd'.
            fp_time (str | list): Specifies the time range(s) in the format 'hhmm_hhmm'.
            cache_dir (str, optional): Directory where downloaded files will be cached. Defaults to
            ".cache".
        """
        self.line_threshold = line_threshold
        self.time_interval = time_interval
        self.traffic_light_speed_threshold = traffic_light_speed_threshold
        self.test_row_numbers = test_row_numbers
        self.params = params
        self.geo_loader = geo_loader
        self.base_url = "https://open-traffic.epfl.ch/wp-content/uploads/mydownloads.php"
        self.fp_location = [fp_location] if isinstance(fp_location, str) else fp_location
        self.fp_date = [fp_date] if isinstance(fp_date, str) else fp_date
        self.fp_time = [fp_time] if isinstance(fp_time, str) else fp_time

        # Dicts:
        self.files_dict = {}
        self.current_file_running = {}
        self.density_exit_entry_files_dict = {}
        self.traffic_light_status_file_dict = {}
        self.test_files = defaultdict(list)
        self.traffic_light_status_dict = {}
        self.cell_vector_occupancy_or_density_dict = {}
        self.link_cumulative_counts_file = {}
        self.cell_entries_dict = {}
        self.first_cell_inflow_dict = {}
        self.next_timestamp_occupancy_dict = {}
        self.cumulative_counts_dict = {}
        self.cell_exits_dict = {}
        self.tasks = {}
        self.exit_cells_files_dict = {}
        self.exit_links_files_dict = {}
        self.ltm_epsilon = None
        self.temp_df = None
        self.df = pl.DataFrame({})
        self._validate_inputs()
        self._download_all_files()
        self._load_dataframe()

    def _validate_inputs(self):
        """
        Validates the input attributes of the class instance.
        """
        for location in self.fp_location:
            if not (location.startswith("d") and (location[1:].isdigit() or location == "dX")):
                raise ValueError(f"Invalid fp_location: {location}")
        for date in self.fp_date:
            if len(date) != 8 or not date.isdigit():
                raise ValueError(f"Invalid fp_date format: {date}")
        for time in self.fp_time:
            if not (time.endswith("00") or time.endswith("30")):
                raise ValueError(f"Invalid fp_time format: {time}")

    def _build_payload(self, location, date, time) -> str:
        """
        Constructs a payload string with file processing location, date, and time.
        """
        return f"fpLocation={location}&fpDate={date}&fpTime={time}"

    def _get_download_url(self, location, date, time) -> str:
        """
        Constructs and returns the full download URL.
        """
        return f"{self.base_url}?{self._build_payload(location, date, time)}"

    def _get_filename(self, location, date, time) -> str:
        """
        Constructs and returns the filename.
        """
        return f"{location}_{date}_{time}"

    def get_cached_filepath(self, location, date, time) -> str:
        """
        Constructs and returns the full file path for a cached file.
        """
        filename = self._get_filename(location, date, time) + ".csv"
        return os.path.join(self.params.cache_dir, filename)

    def check_file_exists_in_cache(self, location, date, time) -> bool:
        """
        Check if the cached file exists in the specified location.
        """
        return os.path.isfile(self.get_cached_filepath(location, date, time))

    def check_file_is_correct(self, downloaded_loc):
        """
        Check if the file is not corrupted.
        """
        with open(downloaded_loc, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line == "Empty Dataset":
                os.remove(downloaded_loc)
                return False
        return True

    def _download_file(self, location, date, time) -> str:
        """
        Downloads a file from a specified URL and saves it to a local cache.
        """
        if self.check_file_exists_in_cache(location, date, time):
            return self.get_cached_filepath(location, date, time)

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Referer": "https://open-traffic.epfl.ch/index.php/downloads/",
            "Origin": "https://open-traffic.epfl.ch",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "*/*"
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            data=self._build_payload(location, date, time),
            stream=True,
            timeout=10  # Set timeout to 10 seconds
        )

        if response.status_code == 200:
            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 8192

            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=self._get_filename(location, date, time),
                leave=False,
                dynamic_ncols=True
            )

            with open(self.get_cached_filepath(location, date, time), "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()

            return self.get_cached_filepath(location, date, time)
        else:
            raise RuntimeError(f"Failed to download file: {response.status_code} - {response.text}")

    def _download_all_files(self):
        """
        Downloads all files for the specified combinations of location, date, and time.
        """
        total_files = len(self.fp_location) * len(self.fp_date) * len(self.fp_time)
        with tqdm(total=total_files, desc="Loading the files", position=0, leave=True) as pbar:
            for location in self.fp_location:
                for date in self.fp_date:
                    for time in self.fp_time:
                        # Use tuple instead of set as dictionary key
                        raw_data_file_path = self._download_file(location, date, time)
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(raw_data_file_path,
                            location,
                            date,
                            time,
                            what_test="raw_data_file_path")
                        )

                        exploded_file_address = self._explode_dataset(
                            raw_data_file_path, location, date, time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(exploded_file_address,
                            location,
                            date,
                            time,
                            what_test="exploded_file_address")
                        )
                        processed_file_address = self._process_link_cell(
                            exploded_file_address, location, date, time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(processed_file_address,
                            location,
                            date,
                            time,
                            what_test="processed_file_address")
                        )
                        vehicle_on_corridor = self._get_vehicle_on_corridor_df(
                            processed_file_address, location, date, time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(vehicle_on_corridor,
                            location, date, time,
                            what_test="vehicle_on_corridor")
                        )
                        removed_vehicles_on_minor_roads = self._fully_process_vehicles(
                            vehicle_on_corridor,
                            location,
                            date,
                            time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(
                                removed_vehicles_on_minor_roads,
                                location,
                                date,
                                time,
                                what_test="removed_vehicles_on_minor_roads"
                            )
                        )
                        self.files_dict[
                            (location, date, time)
                        ] = removed_vehicles_on_minor_roads

                        self.density_exit_entry_files_dict[
                            (location, date, time)
                        ] = self._write_density_entry_exit_df(
                            removed_vehicles_on_minor_roads, location, date, time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(
                                self.density_exit_entry_files_dict[(location, date, time)],
                                location,
                                date,
                                time,
                                what_test="density_entry_exit"
                            )
                        )
                        self.link_cumulative_counts_file[
                            (location, date, time)
                        ] = self.get_cummulative_counts_file(
                            location, date, time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(
                                self.link_cumulative_counts_file[(location, date, time)],
                                location,
                                date,
                                time,
                                what_test="link_cumulative_counts"
                            )
                        )
                        unprocessed_traffic_file = self._get_traffic_light_status(
                            removed_vehicles_on_minor_roads, location, date, time
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(unprocessed_traffic_file,
                            location,
                            date,
                            time,
                            what_test="unprocessed_traffic_light_status")
                        )
                        # _get_processed_traffic_light_status
                        self.traffic_light_status_file_dict[(location, date, time)] = (
                            self._get_processed_traffic_light_status(
                                unprocessed_traffic_file, location, date, time
                            )
                        )
                        self.test_files[(location, date, time)].append(
                            self._get_test_df(
                                self.traffic_light_status_file_dict[(location, date, time)],
                                location, date, time,
                                what_test="processed_traffic_light_status"
                            )
                        )

                        self.exit_cells_files_dict[
                            (location, date, time)
                        ] = self.get_exit_cell(
                            location, date, time
                        )
                        # self.test_files[(location, date, time)].append(
                        #     self._get_test_df(
                        #         self.exit_cells_files_dict[(location, date, time)],
                        #         location, date, time,
                        #         what_test="exit_cells"
                        #     )
                        # )

                        self.exit_links_files_dict[
                            (location, date, time)
                        ] = self.get_exit_link(
                            location, date, time
                        )
                        # self.test_files[(location, date, time)].append(
                        #     self._get_test_df(
                        #         self.exit_links_files_dict[(location, date, time)],
                        #         location, date, time,
                        #         what_test="exit_links"
                        #     )
                        # )

                        
                        pbar.update(1)
        self.df.clear()
        self.df = pl.DataFrame({})

    def _get_trajectory_dataframe(self,
                              track_id,
                              veh_type,
                              traveled_d,
                              avg_speed,
                              trajectory):
        if len(trajectory) % 6 != 0:
            start_preview = trajectory[:6]
            end_preview = trajectory[-6:] if len(trajectory) > 6 else []
            raise ValueError(
                f"[Error] Malformed trajectory (len={len(trajectory)}):\n"
                f"  Start: {start_preview}\n"
                f"  End:   {end_preview}"
            )

        data = defaultdict(list)
        for jndex, item in enumerate(trajectory):
            field_index = jndex % 6
            if field_index == 0:
                data["lat"].append(item)
            elif field_index == 1:
                data["lon"].append(item)
            elif field_index == 2:
                data["speed"].append(item)
            elif field_index == 3:
                data["lon_acc"].append(item)
            elif field_index == 4:
                data["lat_acc"].append(item)
            elif field_index == 5:
                data["trajectory_time"].append(str(round(float(item), 2)))
                data["track_id"].append(track_id)
                data["veh_type"].append(veh_type)
                data["traveled_d"].append(traveled_d)
                data["avg_speed"].append(avg_speed)
        return pl.DataFrame(data)

    def prepare_explode_dataset(self, raw_data_location):
        """
        Processes a raw data file and constructs a concatenated Polars DataFrame
        from its contents.

        This method reads a file line by line, skipping the header, and parses
        each line into its components: track ID, vehicle type, traveled distance,
        average speed, and trajectory points. For each line, it generates a
        DataFrame using the `_get_trajectory_dataframe` method and concatenates
        it to a local DataFrame.

        Args:
            raw_data_location (str): The file path to the raw data file to be
            processed.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the processed and
            exploded dataset.
        """
        local_df = pl.DataFrame({})
        with open(raw_data_location, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Processing...")):
                if i != 0:
                    split = line.strip("\n").strip(" ").split("; ")
                    track_id = split[0]
                    veh_type = split[1]
                    traveled_d = split[2]
                    avg_speed = split[3]
                    trajectory = [v.strip(";").strip("") for v in split[4:]]
                    dataframe = self._get_trajectory_dataframe(
                        track_id,
                        veh_type,
                        traveled_d,
                        avg_speed,
                        trajectory
                    )
                    local_df = pl.concat([local_df, dataframe])
        unique_times = local_df["trajectory_time"].unique().sort()
        groups = local_df.group_by("track_id")
        no_track_ids = groups.n_unique().height
        completed_df = pl.DataFrame({})
        for name, group in tqdm(groups, total=no_track_ids, desc="Adding the timestamp that doesnt exist"):
            max_time_in_group = group["trajectory_time"].max()
            min_time_in_group = group["trajectory_time"].min()
            group_times = group["trajectory_time"]
            group_time_set = set(group_times)
            valid_global_times = [
                t for t in unique_times 
                if min_time_in_group <= t <= max_time_in_group
            ]
            missing_times = sorted(set(valid_global_times) - group_time_set)
            if not missing_times:
                continue
            missing_times = np.array(missing_times, dtype=float)
            latitudes = group["lat"]
            longitudes = group["lon"]
            acc_lat = group["lat_acc"]
            acc_lon = group["lon_acc"]
            speeds = group["speed"]

            
            track_id = str(group["track_id"][0])
            veh_type = str(group["veh_type"][0])
            traveled_d = str(group["traveled_d"][0])
            avg_speed = str(group["avg_speed"][0])
            times_np = np.array(group_times, dtype=float)
            lats_np = np.array(latitudes, dtype=float)
            lons_np = np.array(longitudes, dtype=float)
            acc_lat_np = np.array(acc_lat, dtype=float)
            acc_lon_np = np.array(acc_lon, dtype=float)
            speeds_np = np.array(speeds, dtype=float)
            interp_lats = np.interp(missing_times, times_np, lats_np)
            interp_lons = np.interp(missing_times, times_np, lons_np)
            interp_speeds = np.interp(missing_times, times_np, speeds_np)
            interp_lat_accs = np.interp(missing_times, times_np, acc_lat_np)
            interp_lon_accs = np.interp(missing_times, times_np, acc_lon_np)
            interpolated_rows = [
                {
                    "lat": str(lat),
                    "lon": str(lon),
                    "speed": str(speed),
                    "lon_acc": str(lon_acc),
                    "lat_acc": str(lat_acc),
                    "trajectory_time": str(time),
                    "track_id": str(track_id),
                    "veh_type": str(veh_type),
                    "traveled_d": str(traveled_d),
                    "avg_speed": str(avg_speed),
                }
                for lat, lon, speed, lon_acc, lat_acc, time in zip(
                    interp_lats, interp_lons, interp_speeds,
                    interp_lon_accs, interp_lat_accs, missing_times
                )
            ]
            completed_group = pl.concat([group, pl.DataFrame(interpolated_rows)])
            completed_df = pl.concat([completed_df, completed_group])
        return completed_df

    def _explode_dataset(self, raw_data_location, location, date, time):
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) + "_exploded.csv"
        )
        if os.path.isfile(file_address):
            return file_address
        local_df = self.prepare_explode_dataset(raw_data_location)
        local_df.write_csv(file_address)
        return file_address

    def _load_dataframe(self):
        """
        Loads and concatenates multiple CSV files into a single DataFrame.

        This method iterates over a list of file metadata (`self.files_dict`),
        where each entry contains
        a tuple of (location, date, time) and the corresponding file address.
        It reads each CSV file, adds metadata columns (`location`, `data`, and `time`),
        and concatenates the resulting DataFrame
        to `self.df`.
        Returns:
            polars.DataFrame: A concatenated DataFrame containing data from all the CSV files, 
            with additional metadata columns.
        """
        dataframes = []
        for (_, _, _), file_address in self.files_dict.items():
            read_csv = pl.read_csv(file_address)
            dataframes.append(read_csv)
        if dataframes:
            self.df = pl.concat(dataframes)
        return self.df

    def _process_link_cell(self, exploded_file_address, location, date, time):
        """
        Processes the link and cell data from the DataFrame.
        This method is responsible for loading the geospatial data
        and finding the closest links and cells for each point in the DataFrame.
        """
        processed_file_path = (
            os.path.join(
            self.params.cache_dir,
            f"{self._get_filename(location, date, time)}_withlinkcell_"
            f"{self.geo_loader.get_hash_str()}.csv"
            )
        )
        if os.path.isfile(processed_file_path):
            return (
                processed_file_path
            )

        df = self._find_links_cells(exploded_file_address)
        df.write_csv(processed_file_path)
        return processed_file_path

    def _find_links_cells(self, exploded_file_address):
        """
        Finds the links in the DataFrame and assigns them to the GeoLoader.
        """
        raw_df = pl.read_csv(exploded_file_address)
        if raw_df.is_empty():
            raise ValueError("DataFrame is empty. Cannot find links.")
        length = raw_df.shape[0]
        points = [
            POINT(row["lon"], row["lat"])
            for row in tqdm(raw_df.iter_rows(named=True), total=length, desc="Creating points")
        ]
        batch_size = 50000
        closests_links_cells = []

        with Pool(processes=int(cpu_count() / 2)) as pool:
            for batch in tqdm(
                chunked(points, batch_size),
                total=(len(points) // batch_size) + 1,
                desc="Finding closest cells and links"
            ):
                results = pool.map(self.geo_loader.find_closest_link_and_cell, batch)
                closests_links_cells.extend(results)
        link_ids = []
        link_distances = []
        cell_ids = []
        cell_distances = []
        for (link, link_distance, cell, cell_distance) in closests_links_cells:
            if not isinstance(link_distance, Units.Quantity):
                raise ValueError("Link distance is not a valid Units.M object.")
            if not isinstance(cell_distance, Units.Quantity):
                raise ValueError("Cell distance is not a valid Units.M object.")
            link_ids.append(link.link_id)
            link_distances.append(link_distance.to(Units.M).value)
            cell_ids.append(cell.cell_id)
            cell_distances.append(cell_distance.to(Units.M).value)

        raw_df = raw_df.with_columns([
            pl.Series("link_id", link_ids),
            pl.Series("distance_from_link", link_distances),
            pl.Series("cell_id", cell_ids),
            pl.Series("distance_from_cell", cell_distances)
        ])
        return raw_df

    def _get_vehicle_on_corridor_df(self, with_link_cell_address, location, date, time):
        """
        Filters the DataFrame to include only vehicles on the corridor.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_vehicle_on_corridor_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address
        wlc_df = pl.read_csv(with_link_cell_address)
        wlc_df = wlc_df.filter(pl.col("distance_from_link") < self.line_threshold)
        wlc_df.write_csv(file_address)
        return file_address

    def _fully_process_vehicles(self, vehicle_on_corridor, location, date, time):
        """
        Removes vehicles that are on minor roads from the DataFrame.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_fully_process_vehicles_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address

        wlc_df = pl.read_csv(vehicle_on_corridor)
        groups = wlc_df.group_by("track_id")
        removed_ids = []
        length = wlc_df["track_id"].n_unique()
        for name, group in tqdm(groups, total=length, desc="Removing vehicles on minor roads"):
            if len(group) > 1:
                lon = group["lon"].to_numpy()
                lat = group["lat"].to_numpy()
                reg = LinearRegression()
                reg.fit(lon.reshape(-1, 1), lat)
                if reg.coef_[0] > 0.5:
                    removed_ids.append(name[0] if isinstance(name, (list, tuple)) else name)
        groups = wlc_df.group_by("link_id").agg(
            track_id=pl.col("track_id").unique().alias("track_id")
        )
        intersection = set.intersection(*[set(list_of_vehicle) for list_of_vehicle in groups["track_id"]])
        
        # remove the vehicles that are in the first frame
        min_time = wlc_df["trajectory_time"].min()
        first_frame_veh = wlc_df.filter(pl.col("trajectory_time") == min_time)["track_id"].unique()


        wlc_df = wlc_df.filter(~pl.col("track_id").is_in(removed_ids))
        wlc_df = wlc_df.filter(
            pl.col("track_id").is_in(intersection)
        )
        wlc_df = wlc_df.filter(
            ~pl.col("track_id").is_in(first_frame_veh)
        )
        wlc_df.write_csv(file_address)
        return file_address

    def get_density_entry_exit_df(self, wlc_df):
        # nbbi: Needs test.
        """
        Computes vehicle entry, exit, and density statistics for each cell and time interval
        from a processed trajectory CSV file.

        This method reads a CSV file containing vehicle trajectory data, groups the data by
        link, cell, and time, and calculates:
          - The list of vehicle IDs present in each cell at each time interval.
          - The number of vehicles entering and exiting each cell at each time interval.
          - The number of vehicles present ("on_cell") in each cell at each time interval.
          - The normalized density of vehicles in each cell, based on geometric information.

        Missing timestamps for each (link_id, cell_id) group are filled to ensure continuity.
        Entry and exit events are determined by comparing vehicle lists between consecutive
        time intervals.

        Args:
            fully_processed_file_address (str): Path to the CSV file containing fully processed
            trajectory data.

        Returns:
            pl.DataFrame: A DataFrame containing the following columns:
                - link_id: The ID of the link.
                - cell_id: The ID of the cell.
                - trajectory_time: The time interval.
                - vehicle_ids: List of vehicle IDs present in the cell at the time.
                - entry_count: Number of vehicles entering the cell.
                - exit_count: Number of vehicles exiting the cell.
                - on_cell: Number of vehicles present in the cell.
                - density: Normalized density of vehicles in the cell.
                
        Note:
            Requires `self.time_interval` and `self.geo_loader` to be defined in the class.
            Assumes the existence of a `fill_missing_timestamps` function and that
            `self.geo_loader.links` provides geometric information for normalization.
        """
        # nbbi: Needs test.
        min_time = wlc_df["trajectory_time"].min()
        max_time = wlc_df["trajectory_time"].max()
        counts = wlc_df.group_by(["link_id", "cell_id", "trajectory_time"]).agg([
            pl.col("track_id").alias("vehicle_ids")
        ])
        counts = counts.sort(["link_id", "cell_id", "trajectory_time"])
        
        complete_counts = pl.DataFrame({})
        groups = counts.group_by(["link_id", "cell_id"])
        num_groups = wlc_df.select(["link_id", "cell_id"]).unique().height
        # for _, group in tqdm(groups, total=num_groups, desc="Calculating density"):
            
        for _, group in tqdm(groups, total=num_groups, desc="Calculating density"):
            link_id = group["link_id"].unique()[0]
            cell_id = group["cell_id"].unique()[0]
            # Printing type of trajectory_time
            

            
            # Get previous list of vehicles
            group = group.with_columns([
                pl.col("vehicle_ids").shift(1).alias("prev_vehicles"),
                pl.col("vehicle_ids").shift(-1).alias("next_vehicles")
            ])
            # Fill null with []
            group = group.with_columns([
                
                pl.col("prev_vehicles").fill_null([]),  # sets default to empty list
                pl.col("next_vehicles").fill_null([])  # sets default to empty list
            ])
            
            # Make the prev and next vehicles set instead of list
            group = group.with_columns([
                pl.struct(["vehicle_ids", "prev_vehicles"])
                .map_elements(lambda s: list(set(s["vehicle_ids"]) - set(s["prev_vehicles"])),
                              return_dtype=pl.List(pl.Int64))
                .alias("entries"),

                pl.struct(["vehicle_ids", "next_vehicles"])
                .map_elements(lambda s: list(set(s["vehicle_ids"]) - set(s["next_vehicles"])),
                              return_dtype=pl.List(pl.Int64))
                .alias("exits")
            ])
            group = group.with_columns([
                pl.col("entries").list.len().alias("entry_count"),
                pl.col("exits").list.len().alias("exit_count"),
                pl.col("vehicle_ids").list.len().alias("on_cell"),
            ])

            group = group.select([
                pl.col("link_id"),
                pl.col("cell_id"),
                pl.col("trajectory_time"),
                pl.col("entry_count"),
                pl.col("exit_count"),
                pl.col("on_cell"),
            ])
            filled_group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time,
                max_time
            )
            filled_group = filled_group.with_columns([
                pl.col("link_id").fill_null(link_id),
                pl.col("cell_id").fill_null(cell_id),
                pl.col("on_cell").fill_null(0),
                pl.col("entry_count").fill_null(0),
                pl.col("exit_count").fill_null(0),
            ])


            filled_group = filled_group.with_columns([
                (
                    pl.col("on_cell") / 
                    self.geo_loader.links[link_id].get_cell_length(cell_id).value
                ).alias("density")
            ])
            
            complete_counts = pl.concat([complete_counts, filled_group])
        return complete_counts

    def _write_density_entry_exit_df(self, fully_processed_file_address, location, date, time):
        """
        The fully addressed file is the one that has been exploded, processed, and filtered
        which refers to the file address _fully_process_vehicles returned.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time)
            + "_density_entry_exit_" + self.geo_loader.get_hash_str() + ".csv"
        )

        if os.path.isfile(file_address):
            return file_address
        wlc_df = pl.read_csv(fully_processed_file_address)
        complete_counts = self.get_density_entry_exit_df(wlc_df)
        complete_counts.write_csv(file_address)
        # logger.debug(f"Density DataFrame saved to {file_address}")
        return file_address

    def get_exit_cell(self, location, date, time):
        """
        Returns the outflow cell for each link.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_cell_exit_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address
        density_entry_exit_files = self.density_exit_entry_files_dict[(location, date, time)]
        if not os.path.isfile(density_entry_exit_files):
            raise FileNotFoundError(
                f"Density entry exit file not found for {location}, {date}, {time}"
            )
        df = pl.read_csv(density_entry_exit_files)
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Expected a DataFrame")
        output_data = {}
        for row in df.iter_rows(named=True):
            link_id = row["link_id"]
            cell_id = row["cell_id"]
            exit_value = row["exit_count"]
            trajectory_time = row["trajectory_time"]
            if not isinstance(trajectory_time, float):
                trajectory_time = float(trajectory_time)
            trajectory_time = round(trajectory_time, 2)
            
            if link_id not in output_data:
                output_data[link_id] = {}
            if trajectory_time not in output_data[link_id]:
                output_data[link_id][trajectory_time] = {}
            output_data[link_id][trajectory_time][cell_id] = exit_value
            
        with open(file_address, "w") as f:
            json.dump(output_data, f, indent=4)
        
        return file_address

    def get_exit_link(self, location, date, time):
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_link_exit_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address
        
        cell_exit_file_address = self.get_exit_cell(location, date, time)
        if not isinstance(cell_exit_file_address, str):
            raise Exception("cell_exit_file_address should be a string")
        with open(cell_exit_file_address, "r") as f:
            cell_exit_data = json.load(f)
        
        output_data = {}
        for link_id, times in cell_exit_data.items():
            for time, cells in times.items():
                if link_id not in output_data:
                    output_data[link_id] = {}
                if time not in output_data[link_id]:
                    output_data[link_id][time] = 0
                for _, exit_value in cells.items():
                    output_data[link_id][time] += exit_value

        with open(file_address, "w") as f:
            json.dump(output_data, f, indent=4)
        
        return file_address
    

        
        

    
    def is_vehicle_passed_traffic_light(
        self,
        vehicle_loc: POINT,
        traffic_light_loc: POINT,
    ):
        """
        Determines if a vehicle has passed the specified traffic light location.

        Args:
            vehicle_loc (POINT): The location of the vehicle.
            traffic_light_loc (POINT): The location of the traffic light.

        Returns:
            bool: True if the vehicle has passed the traffic light, False otherwise.
        """
        # This should the vector pointing from the traffic light to the vehicle
        dx = vehicle_loc.x - traffic_light_loc.x
        dy = vehicle_loc.y - traffic_light_loc.y
        rad = atan2(dy, dx)
        theta_deg = (degrees(rad) + 360) % 360  # normalize to [0, 360)
        if 90 <= theta_deg < 180:
            return False
        return True

    def _get_traffic_light_status(self, fully_processed_file_address, location, date, time):
        """
        Returns the traffic light status for the specified location, date, and time.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_traffic_light_status_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address

        # Traffic light locations
        wlc_df = pl.read_csv(fully_processed_file_address)

        points = [
            POINT(row["lon"], row["lat"])
            for row in tqdm(
            wlc_df.iter_rows(named=True),
            total=wlc_df.shape[0],
            desc="Creating points"
            )
        ]
        batch_size = 50000
        closest_locations = []
        with Pool(processes=int(cpu_count() / 2)) as pool:
            for batch in tqdm(
                chunked(points, batch_size),
                total=(len(points) // batch_size) + 1,
                desc="Finding closest traffic lights"
            ):
                results = pool.map(self.geo_loader.find_closest_location, batch)
                closest_locations.extend(results)
        links_id = []
        loc_distances = []
        for (link, loc_distance) in closest_locations:
            if not isinstance(loc_distance, Units.Quantity):
                raise ValueError("Link distance is not a valid Units.M object.")

            links_id.append(link.link_id)
            loc_distances.append(loc_distance.to(Units.M).value)

        wlc_df = wlc_df.with_columns([
            pl.Series("loc_link_id", links_id),
            pl.Series("distance_from_loc_link", loc_distances)
        ])
        # We'll use the same distance threshold as the one we used
        # to find the vehicles on the corridor: self.line_threshold

        wlc_df = wlc_df.filter(pl.col("distance_from_loc_link") < self.line_threshold)
        wlc_df = wlc_df.sort(["link_id", "cell_id", "trajectory_time"])
        groups = wlc_df.group_by(["loc_link_id", "trajectory_time"]).agg([
            pl.col("speed").mean().alias("all_veh_avg_speed")
        ])
        completed_groups = pl.DataFrame({})
        groups = groups.group_by(["loc_link_id"])
        num_groups = wlc_df.select(["loc_link_id"]).unique().height
        min_time = wlc_df["trajectory_time"].min()
        max_time = wlc_df["trajectory_time"].max()

        for link_id, group in tqdm(groups, total=num_groups, desc="Extending the traffic data"):
            link_id = link_id[0] if isinstance(link_id, (list, tuple)) else link_id
            group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time, # type: ignore
                max_time # type: ignore
            )
            group = group.with_columns(
                pl.col("all_veh_avg_speed").fill_null(0.0)
            )
            group = group.with_columns(
                pl.col("loc_link_id").fill_null(link_id)
            )
            completed_groups = pl.concat([completed_groups, group])
        completed_groups.write_csv(file_address)
        # logger.debug(f"Traffic light status DataFram000e00 saved to {file_address}")
        return file_address

    def _get_processed_traffic_light_status(self, unprocessed_traffic_file, location, date, time):
        """
        Returns the traffic status for the specified location, date, and time.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_processed_traffic_light_status_" + self.geo_loader.get_hash_str() + ".csv"
        )

        if os.path.isfile(file_address):
            return file_address

        traffic_df = pl.read_csv(unprocessed_traffic_file)
        green_or_red_column = traffic_df["all_veh_avg_speed"] < self.traffic_light_speed_threshold
        traffic_df = traffic_df.with_columns([
            pl.when(green_or_red_column)
            .then(0)
            .otherwise(1)
            .alias("traffic_light_status")
        ])
        min_time = traffic_df["trajectory_time"].min()
        max_time = traffic_df["trajectory_time"].max()
        completed_traffic_df = pl.DataFrame({})
        groups = traffic_df.group_by(["loc_link_id"])
        num_groups = traffic_df.select(["loc_link_id"]).unique().height
        for link_id, group in tqdm(
            groups,
            total=num_groups,
            desc="Extending the traffic light data"
        ):
            # logger.debug("link_id %s", link_id)
            link_id = link_id[0] if isinstance(link_id, (list, tuple)) else link_id
            group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time, # type: ignore
                max_time # type: ignore
            )
            group = group.with_columns(
                pl.col("traffic_light_status").fill_null(0)
            )
            group = group.with_columns(
                pl.col("loc_link_id").fill_null(link_id)
            )
            completed_traffic_df = pl.concat([completed_traffic_df, group])
        traffic_df = traffic_df.select(["trajectory_time", "traffic_light_status", "loc_link_id"])
        traffic_df.write_csv(file_address)
        return file_address

    def _get_test_df(self, file_location, location, date, time, what_test=""):
        """
        Returns a test DafsafsafsafsafsssstaFrame for the specified file location.

        Handles both CSV and Parquet files. Detects file format and handles list columns
        (for Parquet) gracefully.
        """
        file_ext = os.path.splitext(file_location)[1].lower()
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            f"_test_df_{what_test}_" + self.geo_loader.get_hash_str() +
            (".parquet" if file_ext == ".parquet" else ".csv")
        )
        if os.path.isfile(file_address):
            return file_address

        # Read depending on input format
        if file_ext == ".parquet":
            df = pl.read_parquet(file_location)
            df[:self.test_row_numbers].write_parquet(file_address)
        else:
            try:
                df = pl.read_csv(file_location)
            except pl.exceptions.ComputeError:
                with open(file_location, "rb") as f:
                    raw_data = f.read()
                    detected_encoding = chardet.detect(raw_data)['encoding'] or "ISO-8859-1"
                df = pl.read_csv(file_location, encoding=detected_encoding)
            df[:self.test_row_numbers].write_csv(file_address)
        return file_address

    def get_traffic_light_status_dict(self, location, date, time):
        """
        Returns the traffic light status dictionary for the specified location, date, and time.
        """
        file_address = self.traffic_light_status_file_dict.get((location, date, time), None)
        if file_address is None:
            raise ValueError(f"File not found for {location}, {date}, {time}")

        df = pl.read_csv(file_address)
        tl_dict = {}
        for row in df.iter_rows(named=True):
            trajectory_time = round(row["trajectory_time"], 2)
            traffic_light_status = row["traffic_light_status"]
            loc_link_id = row["loc_link_id"]
            if loc_link_id not in tl_dict:
                tl_dict[loc_link_id] = {}
            tl_dict[loc_link_id][trajectory_time] = traffic_light_status

        return tl_dict

    def get_occupancy_density_flow_entry_exit_dict(self, location, date, time, coi="on_cell"):
        # nbbi: Needs test
        """
        Returns the occupancy or density entry DataFrame for the specified location, date, and time.
        on_cell -> occupancy
        density -> density
        """
        file_address = self.density_exit_entry_files_dict.get((location, date, time), None)
        if file_address is None:
            raise ValueError(f"File not found for {location}, {date}, {time}")

        output_file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            f"_{coi}_" + self.geo_loader.get_hash_str() +  "_" +
            self.params.get_hash_str(["dt"]) + ".json"
        )
        if os.path.isfile(output_file_address):
            with open(output_file_address, "rb") as f:
                data = json.load(f)
            return (convert_keys_to_float(data["cell_vector_occupancy_or_density_dict"]),
            convert_keys_to_float(data["entries_dict"]),
            convert_keys_to_float(data["exits_dict"]))

        df = pl.read_csv(file_address)
        result = (
            df.sort(["link_id", "trajectory_time", "cell_id"])
            .group_by(["link_id", "trajectory_time"])
            .agg([
                pl.col(coi).alias(f"{coi}_vector")
            ])
        )
        cell_vector_occupancy_or_density_dict = {}
        for row in result.iter_rows(named=True):
            if row["link_id"] not in cell_vector_occupancy_or_density_dict:
                cell_vector_occupancy_or_density_dict[row["link_id"]] = {}
            cell_vector_occupancy_or_density_dict[row["link_id"]][
                round(row["trajectory_time"], 2)
            ] = row[f"{coi}_vector"]

        entries_dict = {}
        exits_dict = {}
        for row in df.iter_rows(named=True):
            if row["link_id"] not in entries_dict:
                entries_dict[row["link_id"]] = {}
            if row["link_id"] not in exits_dict:
                exits_dict[row["link_id"]] = {}
            if row["cell_id"] not in entries_dict[row["link_id"]]:
                entries_dict[row["link_id"]][row["cell_id"]] = {}
            if row["cell_id"] not in exits_dict[row["link_id"]]:
                exits_dict[row["link_id"]][row["cell_id"]] = {}

            entries_dict[row["link_id"]][row["cell_id"]][row["trajectory_time"]] = (
                row["entry_count"]
            )
            exits_dict[row["link_id"]][row["cell_id"]][row["trajectory_time"]] = row["exit_count"]
        # Save the data to a JSON file
        with open(output_file_address, "w") as f:
            json.dump({
                "cell_vector_occupancy_or_density_dict": cell_vector_occupancy_or_density_dict,
                "entries_dict": entries_dict,
                "exits_dict": exits_dict
            }, f, indent=4)
        # logger.debug(f"Occupancy or density DataFrame saved to {output_file_address}")
        return cell_vector_occupancy_or_density_dict, entries_dict, exits_dict

    def get_cummulative_counts_file(self, location, date, time):
        # nbbi: Needs test
        """
        Returns the cumulative counts for the specified location, date, and time.
        """
        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_cumulative_counts_" + "_" + self.params.get_hash_str(["dt", "free_flow_speed"]) + "_" +
            self.geo_loader.get_hash_str() + ".csv"
        )
        
        if os.path.isfile(file_address):
            return file_address

        occupanct_exit_entry_df = self.density_exit_entry_files_dict.get(
            (location, date, time), None
        )
        
        if occupanct_exit_entry_df is None:
            raise ValueError(
                f"File not found for {location}, {date}, {time}, "
                "for processing cumulative count"
            )

        occupancy_exit_entry_df = pl.read_csv(occupanct_exit_entry_df)
        groups = occupancy_exit_entry_df.group_by(
            ["link_id", "trajectory_time"]
        )
        group_length = occupancy_exit_entry_df.select(
            ["link_id", "trajectory_time"]
        ).unique().height
        cumulative_counts_data = []
        for name, group in tqdm(groups, total=group_length, desc="Summing entries and exits"):
            # We are only interested in the first and last cell of each link
            link_id, trajectory_time = name[0], name[1]
            trajectory_time = round(float(str(trajectory_time)), 2)

            first_cell_entry = group["entry_count"].sum()
            last_cell_exit = group["exit_count"].sum()
            current_number_of_vehicles = group["on_cell"].sum()
            cumulative_counts_data.append({
                "link_id": link_id,
                "trajectory_time": trajectory_time,
                "first_cell_entry": first_cell_entry,
                "last_cell_exit": last_cell_exit,
                "current_number_of_vehicles": current_number_of_vehicles
            })

        
        cumulative_counts_df = pl.DataFrame(cumulative_counts_data)
        groups = cumulative_counts_df.group_by("link_id")
        min_time = cumulative_counts_df["trajectory_time"].min()
        max_time = cumulative_counts_df["trajectory_time"].max()
        cumulative_cumulative_counts_df = pl.DataFrame({})
        for link_id, group in tqdm(groups, total=group_length, desc="Extending the cumulative counts"):
            link_id = link_id[0] if isinstance(link_id, (list, tuple)) else link_id
            group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time, # type: ignore
                max_time # type: ignore
            )
            group = group.with_columns(
                pl.col("first_cell_entry").fill_null(0.0),
                pl.col("last_cell_exit").fill_null(0.0),
                pl.col("link_id").fill_null(link_id),
                pl.col("current_number_of_vehicles").fill_null(0.0)
            )
            group = group.sort(["trajectory_time"])
            group = group.with_columns([
                pl.col("first_cell_entry").cum_sum().alias("cumulative_link_entry"),
                pl.col("last_cell_exit").cum_sum().alias("cumulative_link_exit")
            ])
            
            cumulative_cumulative_counts_df = pl.concat(
                [cumulative_cumulative_counts_df, group]
            )
        cumulative_cumulative_counts_df = cumulative_cumulative_counts_df.select([
            "link_id",
            "trajectory_time",
            "cumulative_link_entry",
            "cumulative_link_exit",
            "first_cell_entry",
            "last_cell_exit",
            "current_number_of_vehicles"
        ])
        cumulative_cumulative_counts_df = cumulative_cumulative_counts_df.with_columns([
            pl.col("trajectory_time").round(2)
        ])
        cumulative_cumulative_counts_df.write_csv(file_address)
        return file_address

    def activate_tl_status_dict(self, location, date, time):
        """
        Returns the traffic light status dictionary for the specified location, date, and time.
        """
        self.traffic_light_status_dict = self.get_traffic_light_status_dict(location, date, time)

    def activate_occupancy_density_entry_exit_dict(self, location, date, time, coi="on_cell"):
        """
        Returns the density entry DataFrame for the specified location, date, and time.
        """
        self.cell_vector_occupancy_or_density_dict, self.cell_entries_dict, self.cell_exits_dict = (
            self.get_occupancy_density_flow_entry_exit_dict(location, date, time, coi)
        )
    


    def tl_status(self, time, link_id):
        """
        Returns the traffic light status for the specified time and link ID.
        """
        return self.traffic_light_status_dict[link_id].get(time, 0) # Default to 0 if not found

    def get_link_density(self, time, link_id):
        """
        Returns the density for the specified time and link ID.
        """
        return self.cell_vector_occupancy_or_density_dict[link_id][time]

    def get_cell_entry(self, time, link_id, cell_id):
        """
        Returns the entry count for the specified time, link ID, and cell ID.
        """
        return self.cell_entries_dict[link_id][cell_id][time]

    def get_cell_exit(self, time, link_id, cell_id):
        """
        Returns the exit count for the specified time, link ID, and cell ID.
        """
        return self.cell_exits_dict[link_id][cell_id][time]

    def is_tl(self, _):
        """
        Returns True if the specified link ID has a traffic light, False otherwise.
        Right now, all the links are assumed to have traffic lights. nbbi: Later
        write a function to check if the link has a traffic light or not.
        """
        return True

    def get_first_cell_inflow_dict(self, location, date, time):
        """
        Returns the first self inflow for the specified location, date, and time.
        """
        file_address = self.density_exit_entry_files_dict.get((location, date, time), None)
        if file_address is None:
            raise ValueError(f"File not found for {location}, {date}, {time}")
        output_file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_first_cell_inflow_" + self.params.get_hash_str(["dt"]) + "_" +
            self.geo_loader.get_hash_str() + ".json"
        )

        if os.path.isfile(output_file_address):
            with open(output_file_address, "r", encoding="utf-8") as f:
                first_cell_inflow_dict = json.load(f)
            return convert_keys_to_float(first_cell_inflow_dict)

        df = pl.read_csv(file_address)
        df = df.sort(["link_id", "trajectory_time"])
        first_cell_inflow_dict = {}
        groups = df.group_by("link_id", "cell_id")
        num_groups = df.select(["link_id", "cell_id"]).unique().height

        for name, group in tqdm(groups, total=num_groups, desc="Finding first cell inflow"):
            
            link_id = name[0] if isinstance(name, (list, tuple)) else name
            cell_id = name[1] if isinstance(name, (list, tuple)) else name
            if isinstance(link_id, (str, float)):
                link_id = int(link_id)
            if link_id not in first_cell_inflow_dict:
                first_cell_inflow_dict[link_id] = {}
            trajectory_times = group.sort(["trajectory_time"])["trajectory_time"].to_numpy()
            for trajectory_time in trajectory_times:
                if not isinstance(trajectory_time, float):
                    trajectory_time = float(trajectory_time)
                trajectory_time = round(trajectory_time, 2)
                if trajectory_time not in first_cell_inflow_dict[link_id]:
                    first_cell_inflow_dict[link_id][trajectory_time] = {}
                if cell_id not in first_cell_inflow_dict[link_id][trajectory_time]:
                    first_cell_inflow_dict[link_id][trajectory_time][cell_id] = 0
                first_cell_inflow_dict[link_id][trajectory_time][cell_id] += group.filter(
                    (pl.col("trajectory_time") >= trajectory_time) &
                    (pl.col("trajectory_time") < trajectory_time + self.params.dt.to(Units.S).value)
                )["entry_count"].sum()
            
        
        with open(output_file_address, "w", encoding="utf-8") as f:
            json.dump(first_cell_inflow_dict, f, indent=4)
            

        return first_cell_inflow_dict
    
    
    def get_cumulative_count_point_queue_spatial_queue(self, location, date, time):
        """
        Returns the cumulative counts for the specified location, date, and time.
        """
        # nbbi: Needs test
        cumulative_counts_file = self.link_cumulative_counts_file.get((location, date, time), None)
        if not isinstance(cumulative_counts_file, str):
            raise ValueError(f"File not found for {location}, {date}, {time}")
        
        output_file_address = (
            self.params.cache_dir + "/" +
            self._get_filename(location, date, time) +
            "_cumulative_count_dt_ffs_link_length_" +
            self.params.get_hash_str(["dt", "free_flow_speed"]) + "_" +
            self.geo_loader.get_hash_str() + ".json"
        )
        if os.path.isfile(output_file_address):
            with open(output_file_address, "r", encoding="utf-8") as f:
                cumulative_counts_dict = json.load(f)
                
            return convert_keys_to_float(cumulative_counts_dict)

        cumulative_counts_df = pl.read_csv(cumulative_counts_file)
        cumulative_counts_df = self.get_cummulative_counts_based_on_t(
            cumulative_counts_df,
            link_based_t={
                link.link_id: self.params.dt - (link.get_length() / self.params.free_flow_speed) 
                for link_id, link in self.geo_loader.links.items()
            }
        )
        cumulative_counts_dict = {}
        """
        "link_id": [],
            "target_time": [],
            "cummulative_count_upstream_offset": [],
            "trajectory_time": [],
            "cummulative_count_downstream": [],
            "cummulative_count_upstream": [],
            "entry_count": [],
            "current_number_of_vehicles": []
        """
        for row in cumulative_counts_df.iter_rows(named=True):
            if row["link_id"] not in cumulative_counts_dict:
                cumulative_counts_dict[row["link_id"]] = {}
            if row["trajectory_time"] not in cumulative_counts_dict[row["link_id"]]:
                cumulative_counts_dict[row["link_id"]][row["trajectory_time"]] = {}
            cumulative_counts_dict[row["link_id"]][row["trajectory_time"]]["target_time"] = row["target_time"]
            cumulative_counts_dict[row["link_id"]][row["trajectory_time"]]["cummulative_count_upstream_offset"] = row["cummulative_count_upstream_offset"]
            cumulative_counts_dict[row["link_id"]][row["trajectory_time"]]["cummulative_count_downstream"] = row["cummulative_count_downstream"]
            cumulative_counts_dict[row["link_id"]][row["trajectory_time"]]["cummulative_count_upstream"] = row["cummulative_count_upstream"]
            cumulative_counts_dict[row["link_id"]][row["trajectory_time"]]["entry_count"] = row["entry_count"]
            cumulative_counts_dict[row["link_id"]][row["trajectory_time"]]["current_number_of_vehicles"] = row["current_number_of_vehicles"]




        # Convert keys to float
        with open(output_file_address, "w", encoding="utf-8") as f:
            json.dump(cumulative_counts_dict, f, indent=4)
        
        return cumulative_counts_dict
    def _get_cumulative_count_ltm_for_multiprocessing(self, args):
        """
        For multiprocessing LTM.

        Required arguments: x, time, link_id, cell_id, link_length
        """
        # Requires: x, time, link_id, cell_id, link_length
        for arg in ["x", "time", "link_id", "cell_id", "link_length"]:
            if arg not in args:
                raise ValueError(f"Missing argument: {arg}")
        x = args["x"]
        time = args["time"]
        link_id = args["link_id"]
        cell_id = args["cell_id"]
        link_length = args["link_length"]
        
        if self.ltm_epsilon is None:
            raise ValueError("LTM epsilon is not set.")
        if self.temp_df is None:
            raise ValueError("Temp DataFrame is not set.")
        time_units = time * Units.S
        x = x * Units.M
        epsilon_x = self.ltm_epsilon * Units.M
        epsilon_t = self.ltm_epsilon * Units.S
        link_length = link_length * Units.M
        # freeflow with epsilon location
        target_time_freeflow_with_eps_x = time_units + self.params.dt - ((x+epsilon_x)/self.params.free_flow_speed)
        upstream_value_freeflow_with_eps_x, downstream_value_freeflow_with_eps_x = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_time_freeflow_with_eps_x
        )
        # freeflow with epsilon horizon
        target_value_freeflow_with_eps_t = time_units + (self.params.dt + epsilon_t) - (x / self.params.free_flow_speed)
        upstream_value_freeflow_with_eps_t, downstream_value_freeflow_with_eps_t = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_value_freeflow_with_eps_t
        )

        target_time_freeflow = (time_units + self.params.dt - (x / self.params.free_flow_speed))
        upstream_value_freeflow, downstream_value_freeflow = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_time_freeflow
        )

        # freeflow without epsilon
        target_time_freeflow = (time_units + self.params.dt - (x / self.params.free_flow_speed))
        upstream_value_freeflow, downstream_value_freeflow = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_time_freeflow
        )
        

        # wave with epsilon location
        target_time_wavespeed_with_eps_x = (self.params.wave_speed * self.params.dt + (x+epsilon_x) - link_length) / self.params.wave_speed
        upstream_value_wavespeed_with_eps_x, downstream_value_wavespeed_with_eps_x = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_time_wavespeed_with_eps_x
        )
        # wave with epsilon horizon
        target_time_wavespeed_with_eps_t = (self.params.wave_speed * (self.params.dt + epsilon_t) + (x) - link_length) / self.params.wave_speed
        upstream_value_wavespeed_with_eps_t, downstream_value_wavespeed_with_eps_t = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_time_wavespeed_with_eps_t
        )
        # wave without epsilon
        target_time_wavespeed = (self.params.wave_speed * self.params.dt + (x) - link_length) / self.params.wave_speed
        upstream_value_wavespeed, downstream_value_wavespeed = self.get_cummulative_counts(
            self.temp_df[link_id],
            target_time_wavespeed
        )
        return {
            "upstream_value_freeflow_with_eps_x": upstream_value_freeflow_with_eps_x,
            "downstream_value_freeflow_with_eps_x": downstream_value_freeflow_with_eps_x,
            "upstream_value_freeflow_with_eps_t": upstream_value_freeflow_with_eps_t,
            "downstream_value_freeflow_with_eps_t": downstream_value_freeflow_with_eps_t,
            "upstream_value_freeflow": upstream_value_freeflow,
            "downstream_value_freeflow": downstream_value_freeflow,

            "upstream_value_wavespeed_with_eps_x": upstream_value_wavespeed_with_eps_x,
            "downstream_value_wavespeed_with_eps_x": downstream_value_wavespeed_with_eps_x,
            "upstream_value_wavespeed_with_eps_t": upstream_value_wavespeed_with_eps_t,
            "downstream_value_wavespeed_with_eps_t": downstream_value_wavespeed_with_eps_t,
            "upstream_value_wavespeed": upstream_value_wavespeed,
            "downstream_value_wavespeed": downstream_value_wavespeed,
            "cell_id": cell_id,
            "link_id": link_id,
            "trajectory_time": time,
            "x": x.to(Units.M).value,
            "target_time_freeflow": target_time_freeflow.to(Units.S).value,
            "target_time_freeflow_with_eps_x": target_time_freeflow_with_eps_x.to(Units.S).value,
            "target_time_freeflow_with_eps_t": target_value_freeflow_with_eps_t.to(Units.S).value,
            "target_time_wavespeed_with_eps_x": target_time_wavespeed_with_eps_x.to(Units.S).value,
            "target_time_wavespeed_with_eps_t": target_time_wavespeed_with_eps_t.to(Units.S).value,
            "target_time_wavespeed": target_time_wavespeed.to(Units.S).value,
            "link_length": link_length.to(Units.M).value
            
        }
    
    def _cumulative_count_ltm_list_to_dict(self, cumulative_counts_result_list):
        

        cumulative_counts_dict = {}
        for result in cumulative_counts_result_list:
            if result["link_id"] not in cumulative_counts_dict:
                cumulative_counts_dict[result["link_id"]] = {}
            if result["trajectory_time"] not in cumulative_counts_dict[result["link_id"]]:
                cumulative_counts_dict[result["link_id"]][result["trajectory_time"]] = {}
            if result["cell_id"] not in cumulative_counts_dict[result["link_id"]][result["trajectory_time"]]:
                cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]] = {}
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["x"] = result["x"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["link_length"] = result["link_length"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["upstream_value_freeflow_with_eps_x"] = result["upstream_value_freeflow_with_eps_x"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["downstream_value_freeflow_with_eps_x"] = result["downstream_value_freeflow_with_eps_x"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["upstream_value_freeflow_with_eps_t"] = result["upstream_value_freeflow_with_eps_t"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["downstream_value_freeflow_with_eps_t"] = result["downstream_value_freeflow_with_eps_t"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["upstream_value_freeflow"] = result["upstream_value_freeflow"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["downstream_value_freeflow"] = result["downstream_value_freeflow"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["upstream_value_wavespeed_with_eps_x"] = result["upstream_value_wavespeed_with_eps_x"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["downstream_value_wavespeed_with_eps_x"] = result["downstream_value_wavespeed_with_eps_x"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["upstream_value_wavespeed_with_eps_t"] = result["upstream_value_wavespeed_with_eps_t"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["downstream_value_wavespeed_with_eps_t"] = result["downstream_value_wavespeed_with_eps_t"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["upstream_value_wavespeed"] = result["upstream_value_wavespeed"]
            cumulative_counts_dict[result["link_id"]][result["trajectory_time"]][result["cell_id"]]["downstream_value_wavespeed"] = result["downstream_value_wavespeed"]
            
        return cumulative_counts_dict

    def get_cumulative_count_ltm(self, location, date, time, epsilon = 0.01):
        """
        Returns the cumulative counts for the specified location, date, and time.
        """
        # nbbi: Needs test
        # nbbi: This can be more time efficient
        cumulative_counts_file = self.link_cumulative_counts_file.get((location, date, time), None)
        if not isinstance(cumulative_counts_file, str):
            raise ValueError(f"File not found for {location}, {date}, {time}")
        
        output_file_address = (
            self.params.cache_dir + "/" +
            self._get_filename(location, date, time) +
            f"_cumulative_count_ltm_epsilon_{epsilon}_" + self.params.get_hash_str(["dt", "free_flow_speed", "wave_speed"]) + "_" +
            self.geo_loader.get_hash_str() + ".json"
        )
        if os.path.isfile(output_file_address):
            with open(output_file_address, "r", encoding="utf-8") as f:
                cumulative_counts_dict = json.load(f)
            return convert_keys_to_float(cumulative_counts_dict)
        
        cummulative_counts_df = pl.read_csv(cumulative_counts_file)
        cummulative_counts_df = cummulative_counts_df
        args = []
        self.temp_df = {}
        self.ltm_epsilon = epsilon
        for link_id, link in tqdm(self.geo_loader.links.items(), desc="Preparing args for", total=len(self.geo_loader.links)):
            link_df = cummulative_counts_df.filter(
                pl.col("link_id") == link_id
            ).sort(["trajectory_time"])
            self.temp_df[link_id] = link_df
            trajectory_time = link_df["trajectory_time"].unique().to_numpy()
            for raw_time in tqdm(trajectory_time, total=len(trajectory_time), desc=f"Preparing args for for link {link_id}"):
                x = 0
                for cell_id, cell in link.cells.items():
                    x += cell.get_length().to(Units.M).value
                    args.append({
                        "x": x,
                        "time": raw_time,
                        "link_id": link_id,
                        "cell_id": cell_id,
                        "link_length": link.get_length().to(Units.M).value
                    })
        # Multiprocessing with batch
        batch_size = 10000
        cumulative_counts_result = []
        with Pool(processes=int(cpu_count() / 2)) as pool:
            for batch in tqdm(
                chunked(args, batch_size),
                total=(len(args) // batch_size) + 1,
                desc="Finding cumulative counts for LTM"
            ):
                results = pool.map(self._get_cumulative_count_ltm_for_multiprocessing, batch)
                cumulative_counts_result.extend(results)
        
        
        cumulative_counts_dict = self._cumulative_count_ltm_list_to_dict(cumulative_counts_result)
        with open(output_file_address, "w", encoding="utf-8") as f:
            json.dump(cumulative_counts_dict, f, indent=4)
        return cumulative_counts_dict

    def get_next_cell_exit_file(self, location, date, time):
        """
        Get the next cell exit for the specified location, date, and time.
        """
        output_file_address = (
            self.params.cache_dir + "/" +
            self._get_filename(location, date, time) +
            "_next_cell_exit_" +
            self.params.get_hash_str(["dt"]) + "_" +
            self.geo_loader.get_hash_str() + ".json"
        )
        if os.path.isfile(output_file_address):
            return output_file_address
        
        density_entry_exit_files = self.density_exit_entry_files_dict[(location, date, time)]
        if not os.path.isfile(density_entry_exit_files):
            raise FileNotFoundError(
                f"Density entry exit file not found for {location}, {date}, {time}"
            )
        
        df = pl.read_csv(density_entry_exit_files)

        groups = df.group_by(["link_id", "cell_id"])
        ground_truth_occupancy = {}
        dt_seconds = self.params.dt.to(Units.S).value
        min_time = df["trajectory_time"].min()
        max_time = df["trajectory_time"].max()
        num_groups = df.select(["link_id", "cell_id"]).unique().height
        for name, group in tqdm(groups, desc="Finding next timestamp cell exit", total=num_groups):
            link_id, cell_id = name[0], name[1]
            if link_id not in ground_truth_occupancy:
                ground_truth_occupancy[link_id] = {}
            group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time, # type: ignore
                max_time # type: ignore
            )
            group = group.with_columns(
                pl.col("exit_count").fill_null(0.0)
            )
            # Round the group trajectory_time to 2 decimal places
            group = group.with_columns(
                pl.col("trajectory_time").cast(pl.Float64).round(2)
            )

            group = group.sort(["trajectory_time"])
            times = group["trajectory_time"].to_numpy()
            for idx, current_time in enumerate(times):
                target_time = current_time + dt_seconds
                next_exit = (
                    group.filter(pl.col("trajectory_time") == target_time)["exit_count"].first()
                    if target_time in group["trajectory_time"] else 0
                ) # nbbi: This part might be causing errors! Also, it's too slow!
                if current_time not in ground_truth_occupancy[link_id]:
                    ground_truth_occupancy[link_id][current_time] = {}
                ground_truth_occupancy[link_id][current_time][cell_id] = next_exit
        with open(output_file_address, "w") as f:
            json.dump(ground_truth_occupancy, f, indent=4)
        return output_file_address
        
    def get_next_link_exit_file(self, location, date, time):
        """
        Get the next link exit for the specified location, date, and time.
        """
        output_file_address = (
            self.params.cache_dir + "/" +
            self._get_filename(location, date, time) +
            "_next_link_exit_" +
            self.params.get_hash_str(["dt"]) + "_" +
            self.geo_loader.get_hash_str() + ".json"
        )
        if os.path.isfile(output_file_address):
            return output_file_address
        
        cell_link_address = self.get_next_cell_exit_file(location, date, time)
        with open(cell_link_address, "r") as f:
            cell_exit_dict = json.load(f)
         
        for link_id, trajectory_time in cell_exit_dict.items():
            for time, exit_counts in trajectory_time.items():
                cell_exit_dict[link_id][time] = sum(exit_counts.values())

        with open(output_file_address, "w") as f:
            json.dump(cell_exit_dict, f, indent=4)
        return output_file_address

    def activate_cumulative_dict_queue(self, location, date, time):
        """
        Returns the cumulative counts for the specified location, date, and time.
        """
        self.cumulative_counts_dict = self.get_cumulative_count_point_queue_spatial_queue(location, date, time)
        

    def activate_first_cell_inflow_dict(self, location, date, time):
        """
        Returns the first cell inflow dictionary for the specified location, date, and time.
        """
        self.first_cell_inflow_dict = self.get_first_cell_inflow_dict(location, date, time)

    def activate_next_timestamp_occupancy(self, location, date, time):
        """
        Returns the next timestamp occupancy for the specified location, date, and time.
        """
        self.next_timestamp_occupancy_dict = self.get_next_timestamp_occupancy(location, date, time)
    
    def activate_cummulative_counts_ltm(self, location, date, time):
        """
        Returns the cumulative counts for the specified location, date, and time.
        """
        self.cumulative_counts_dict = self.get_cumulative_count_ltm(location, date, time)

    def activate_exit_cell(self, location, date, time):
        """
        Sets the exit cell values
        """
        exit_cell_file = self.exit_cells_files_dict.get((location, date, time), None)
        if exit_cell_file is None or not os.path.isfile(exit_cell_file):
            raise Exception(f"Exit file for {location}, {date}, {time} ")
        with open(exit_cell_file, "r") as f:
            self.exit_cells_values = json.load(f)
    
    def activate_exit_link(self, location, date, time):
        """
        Sets the exit link values
        """
        exit_link_file = self.exit_links_files_dict.get((location, date, time), None)
        if exit_link_file is None or not os.path.isfile(exit_link_file):
            raise Exception(f"Exit file for {location}, {date}, {time} ")
        with open(exit_link_file, "r") as f:
            self.exit_links_values = json.load(f)
        
    
    def activate_next_exit_cell(self, location, date, time):
        """
        Sets the next exit cell values
        """
        next_exit_cell_file = self.get_next_cell_exit_file(location, date, time)
        if next_exit_cell_file is None or not os.path.isfile(next_exit_cell_file):
            raise Exception(f"Next exit cell file for {location}, {date}, {time} not found")
        with open(next_exit_cell_file, "r") as f:
            self.next_exit_cell_values = convert_keys_to_float(json.load(f))
    
    def activate_next_exit_link(self, location, date, time):
        """
        Sets the next exit link values
        """
        next_exit_link_file = self.get_next_link_exit_file(location, date, time)
        if next_exit_link_file is None or not os.path.isfile(next_exit_link_file):
            raise Exception(f"Next exit link file for {location}, {date}, {time} not found")
        with open(next_exit_link_file, "r") as f:
            self.next_exit_link_values = convert_keys_to_float(json.load(f))

    def set_params(self, params: Parameters):
        """
        Set the parameters for the traffic model.
        
        Args:
            params (Parameters): Parameters object containing traffic model parameters.
        """
        self.params = params
    
    def get_next_timestamp_occupancy(self, location, date, time):
        """
        Returns the occupancy of the next timestamp for the specified location, date, and time.
        """
        # nbbi: Needs test

        file_address = (
            self.params.cache_dir + "/" + self._get_filename(location, date, time) +
            "_next_timestamp_occupancy_" + self.geo_loader.get_hash_str()  + "_" +
           self.params.get_hash_str(['dt']) + ".json"
        )
        if os.path.isfile(file_address):
            with open(file_address, "r", encoding="utf-8") as f:
                next_timestamp_occupancy_dict = json.load(f)
            return convert_keys_to_float(next_timestamp_occupancy_dict)
        
        occupancy_df = self.density_exit_entry_files_dict.get(
            (location, date, time), None
        )
        if occupancy_df is None:
            raise ValueError(
                f"Occupancy file not found for {location}, {date}, {time}, "
                "for processing occupancy"
            )

        occupancy_df = pl.read_csv(occupancy_df)
        # For each timestamp (not timestep), get the occupancy of the next timestep(not timestamp)
        # Timestep is self.params.dt
        # Find the closest timestamp to the next timestep in the same link.
        occupancy_df = occupancy_df.sort(["link_id", "trajectory_time"])
        min_time = occupancy_df["trajectory_time"].min()
        max_time = occupancy_df["trajectory_time"].max()
        groups = occupancy_df.group_by(["link_id", "cell_id"])
        ground_truth_occupancy = []
        dt_seconds = self.params.dt.to(Units.S).value
        num_groups = occupancy_df.select(["link_id", "cell_id"]).unique().height
        for name, group in tqdm(groups, desc="Finding next timestamp occupancy", total=num_groups):
            link_id, cell_id = name[0], name[1]
            group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time, # type: ignore
                max_time # type: ignore
            )
            group = group.with_columns(
                pl.col("on_cell").fill_null(0.0)
            )
            # Round the group trajectory_time to 2 decimal places
            group = group.with_columns(
                pl.col("trajectory_time").cast(pl.Float64).round(2)
            )

            group = group.sort(["trajectory_time"])
            times = group["trajectory_time"].to_numpy()
            occupancies = group["on_cell"].to_numpy()
            for idx, current_time in enumerate(times):
                target_time = current_time + dt_seconds
                next_occupancy = (
                    group.filter(pl.col("trajectory_time") == target_time)["on_cell"].first()
                    if target_time in group["trajectory_time"] else 0
                ) # nbbi: This part might be causing errors! Also, it's too slow!

                ground_truth_occupancy.append({
                    "link_id": link_id,
                    "trajectory_time": current_time,
                    "on_cell_now": occupancies[idx],
                    "on_cell_next": next_occupancy,
                    "target_time": target_time,
                    "cell_id": cell_id
                })

        ground_truth_occupancy_df = pl.DataFrame(ground_truth_occupancy)
        result = (
            ground_truth_occupancy_df.sort(["link_id", "trajectory_time", "cell_id"])
            .group_by(["link_id", "trajectory_time"])
            .agg([
                pl.col("on_cell_now").alias(f"on_cell_vector"),
                pl.col("on_cell_next").alias(f"on_cell_next_vector")
            ])
        )
        occcupancy_ground_truths = {}
        for row in result.iter_rows(named=True):
            if row["link_id"] not in occcupancy_ground_truths:
                occcupancy_ground_truths[row["link_id"]] = {}
            occcupancy_ground_truths[row["link_id"]][
                round(row["trajectory_time"], 2)
            ] = {"current_occupancy": row["on_cell_vector"], "next_occupancy": row["on_cell_next_vector"]}
        # Save the data to a JSON file
        with open(file_address, "w", encoding="utf-8") as f:
            json.dump(occcupancy_ground_truths, f, indent=4)
        return occcupancy_ground_truths
    def get_cummulative_counts(self,
                               link_df: pl.DataFrame,
                               target_time: Units.Quantity,
                               ):
        """
        For Point Queue and Spatial Queue.
        """
        # nbbi: Needs test
        if not isinstance(target_time, Units.Quantity):
            raise ValueError(f"target_time should be a Units.Quantity object. Got {type(target_time)}")
        target_time_value = target_time.to(Units.S).value
        link_df_unique_links = link_df.select("link_id").unique()
        if len(link_df_unique_links) != 1:
            raise ValueError(f"Link ID should be unique. Got {link_df_unique_links}")
        
        all_trajectory_times = link_df["trajectory_time"].to_numpy()
        closest_index = np.searchsorted(all_trajectory_times, target_time_value)
        if closest_index == len(all_trajectory_times) - 1 or closest_index == len(all_trajectory_times):
            cummulative_count_upstream = 0.0
            cummulative_count_downstream = 0.0
        else:
            cummulative_count_upstream = link_df.filter(
                pl.col("trajectory_time") == all_trajectory_times[closest_index]
            )["cumulative_link_entry"].first()
            cummulative_count_downstream = link_df.filter(
                pl.col("trajectory_time") == all_trajectory_times[closest_index]
            )["cumulative_link_exit"].first()
        
        return cummulative_count_upstream, cummulative_count_downstream

        
    def get_cummulative_counts_based_on_t(self, cumulative_counts_df: pl.DataFrame, t: Optional[Units.Quantity] = None, link_based_t: Optional[dict] = None) -> pl.DataFrame:
        """
        For Point Queue and Spatial Queue.
        """
        # nbbi: Needs test
        if t is not None:
            if not isinstance(t, Units.Quantity):
                raise ValueError(f"t should be a Units.Quantity object. Got {type(t)}")

        groups = cumulative_counts_df.group_by("link_id")
        num_groups = cumulative_counts_df.select(["link_id"]).unique().height

        final_cummulative_counts = {
            "link_id": [],
            "target_time": [],
            "cummulative_count_upstream_offset": [],
            "cummulative_count_downstream_offset": [],
            "trajectory_time": [],
            "cummulative_count_downstream": [],
            "cummulative_count_upstream": [],
            "entry_count": [],
            "current_number_of_vehicles": []
            
        }
        for link_id, group in tqdm(groups, total=num_groups, desc="Finding cumulative counts"):
            group = group.with_columns(group["trajectory_time"].cast(pl.Float64).round(2))
            link_id = link_id[0] if isinstance(link_id, (list, tuple)) else link_id
            group = group.sort(["trajectory_time"])
            all_trajectory_times = group["trajectory_time"].to_numpy()
            for row in group.iter_rows(named=True):
                trajectory_time = round(row["trajectory_time"], 2)
                
                if link_based_t != None:
                    if link_id not in link_based_t:
                        raise ValueError(
                            f"Link ID {link_id} not found in link_based_t dictionary."
                        )
                    if not isinstance(link_based_t[link_id], Units.Quantity):
                        raise ValueError(
                            f"Value for link ID {link_id} in link_based_t is not a valid Units.Quantity."
                        )
                    target_time = (trajectory_time * Units.S) + link_based_t[link_id].to(Units.S)
                else:
                    if t is None:
                        raise ValueError("At least one of t or link_based_t should be provided.")
                    target_time = (trajectory_time * Units.S) + t
                target_time = round(target_time.to(Units.S).value, 2)

                if not isinstance(target_time, float):
                    target_time = float(target_time)
                closest_index = np.searchsorted(all_trajectory_times, target_time)
                if closest_index == len(all_trajectory_times) - 1 or closest_index == len(all_trajectory_times):
                    cummulative_count_upstream_offset = 0.0
                    cummulative_count_downstream_offset = 0.0
                else:
                    cummulative_count_upstream_offset = group.filter(
                        pl.col("trajectory_time") == all_trajectory_times[closest_index]
                    )["cumulative_link_entry"].first()
                    cummulative_count_downstream_offset = group.filter(
                        pl.col("trajectory_time") == all_trajectory_times[closest_index]
                    )["cumulative_link_exit"].first()
                
                cumulative_downstream = row["cumulative_link_exit"]


                final_cummulative_counts["link_id"].append(link_id)
                final_cummulative_counts["target_time"].append(target_time)
                final_cummulative_counts["cummulative_count_upstream_offset"].append(
                    cummulative_count_upstream_offset
                )

                final_cummulative_counts["cummulative_count_downstream_offset"].append(
                    cummulative_count_downstream_offset
                )
                final_cummulative_counts["trajectory_time"].append(trajectory_time)
                final_cummulative_counts["cummulative_count_downstream"].append(
                    cumulative_downstream
                )
                final_cummulative_counts["cummulative_count_upstream"].append(
                    row["cumulative_link_entry"]
                )

                final_cummulative_counts["entry_count"].append(row["first_cell_entry"])
                final_cummulative_counts["current_number_of_vehicles"].append(row["current_number_of_vehicles"])

        return pl.DataFrame(final_cummulative_counts)


    def destruct(self):
        """
        Clears the dictionaries and the df.
        """
        self.cell_vector_occupancy_or_density_dict.clear()
        self.first_cell_inflow_dict.clear()
        self.traffic_light_status_dict.clear()
        self.cell_entries_dict.clear()
        self.cell_exits_dict.clear()
        self.df = pl.DataFrame({})


    def prepare_ctm_tasks(self, location, date, time):

        """
        Prepares the dictionaries and the df for further processing.
        """
        self.activate_tl_status_dict(location, date, time)
        self.activate_next_timestamp_occupancy(location, date, time)
        self.activate_first_cell_inflow_dict(location, date, time)
        self.activate_next_exit_cell(location, date, time)

        
        self.current_file_running = {
            "location": location,
            "date": date,
            "time": time
        }
        file_address = (
            self.params.cache_dir + "/" +
            f"{self._get_filename(location, date, time)}_prepared_ctm_tasks_"
            f"{self.geo_loader.get_hash_str()}_{self.params.get_hash_str(['cache_dir', 'dt', 'alpha', 'jam_density_link', 'q_max'])}.json"
        )
        if os.path.isfile(file_address):
            self.tasks = json.load(open(file_address, "r", encoding="utf-8"))
            return
        
        tasks = [] # ["cell_occupancies list", "first_cell_inflow", "link_id", "is_tl", "tl_status"]
        cell_capacities = self.geo_loader.get_cell_capacities(self.params)
        max_flows = self.geo_loader.get_max_flows(self.params)
        for link_id, cell_dict in self.next_timestamp_occupancy_dict.items():
            for trajectory_time, occupancy_list in cell_dict.items():
                tasks.append(
                    {
                        "occupancy_list": occupancy_list["current_occupancy"],
                        "cell_capacities": deepcopy(cell_capacities[link_id]),
                        "next_occupancy": occupancy_list["next_occupancy"],
                        "max_flows": deepcopy(max_flows[link_id]),
                        "inflow": self.first_cell_inflow_dict[link_id].get(trajectory_time, 0), # nbbi: This part might be causing errors!
                        "link_id": link_id,
                        "is_tl": self.is_tl(link_id),
                        "tl_status": self.tl_status(trajectory_time, link_id),
                        "trajectory_time": trajectory_time,
                        "flow_capacity": self.params.flow_capacity.to(1).value, # type: ignore
                        "alpha": self.params.alpha.to(1).value,
                        "next_exit": list((dict(sorted(self.next_exit_cell_values[link_id][trajectory_time].items(), key= lambda x: x[0]))).values())
                    }
                )
                
        with open(file_address, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=4)
            
        self.tasks = tasks
        self.destruct()
    

    def prepare_pq_tasks(self, location, date, time):
        """
        
        Prepares the necessary tasks for the specified location, date, and time.
        """
        self.activate_cumulative_dict_queue(location, date, time)
        self.activate_next_timestamp_occupancy(location, date, time)
        self.activate_tl_status_dict(location, date, time)
        self.activate_next_exit_link(location, date, time)
        self.activate_first_cell_inflow_dict(location, date, time)

        self.current_file_running = {
            "location": location,
            "date": date,
            "time": time
        }
        logger.debug(
            f"Preparing tasks for {location}, {date}, {time} with params: {self.params}"
        )
        file_address = (
            self.params.cache_dir + "/" +
            f"{self._get_filename(location, date, time)}_prepared_pq_tasks_"
            f"{self.geo_loader.get_hash_str()}_{self.params.get_hash_str(['cache_dir', 'free_flow_speed', 'dt', 'q_max'])}.json"
        )
        if os.path.isfile(file_address):
            self.tasks = json.load(open(file_address, "r", encoding="utf-8"))
            for index in range(len(self.tasks)):
                self.tasks[index]["dt"] = self.tasks[index]["dt"] * Units.S
                self.tasks[index]["q_max_up"] = self.tasks[index]["q_max_up"] * Units.PER_HR
                self.tasks[index]["q_max_down"] = self.tasks[index]["q_max_down"] * Units.PER_HR
            return
        tasks = []
        for link_id, cell_dict in self.cumulative_counts_dict.items(): # type: ignore
            for trajectory_time, data in cell_dict.items():
                tasks.append(
                    {
                        "q_max_up": self.params.q_max,
                        "q_max_down": self.params.q_max,
                        "next_occupancy": sum(self.next_timestamp_occupancy_dict[link_id][trajectory_time]["next_occupancy"]),
                        "cummulative_count_upstream_offset": data["cummulative_count_upstream_offset"],
                        "cummulative_count_downstream": data["cummulative_count_downstream"],
                        "cummulative_count_upstream": data["cummulative_count_upstream"],
                        "current_number_of_vehicles": data["current_number_of_vehicles"],
                        "inflow": self.first_cell_inflow_dict[link_id].get(trajectory_time, 0),
                        "entry_count": data["entry_count"],
                        "dt": self.params.dt,
                        "trajectory_time": trajectory_time,
                        "tl_status": self.tl_status(trajectory_time, link_id),
                        "link_id": link_id,
                        "next_exit": self.next_exit_link_values[link_id][trajectory_time]
                    }
                )

        with open(file_address, "w", encoding="utf-8") as f:
            copy_tasks = deepcopy(tasks)
            for index in range(len(copy_tasks)):
                copy_tasks[index]["dt"] = copy_tasks[index]["dt"].to(Units.S).value
                copy_tasks[index]["q_max_up"] = copy_tasks[index]["q_max_up"].to(Units.PER_HR).value
                copy_tasks[index]["q_max_down"] = copy_tasks[index]["q_max_down"].to(Units.PER_HR).value
            json.dump(copy_tasks, f, indent=4)
        self.tasks = tasks
        self.destruct()
    
    def prepare_sq_tasks(self, location, date, time):
        """
        
        Prepares the necessary tasks for the specified location, date, and time.
        """
        self.activate_cumulative_dict_queue(location, date, time)
        self.activate_tl_status_dict(location, date, time)
        self.activate_next_timestamp_occupancy(location, date, time)
        self.activate_next_exit_link(location, date, time)
        self.activate_first_cell_inflow_dict(location, date, time)
        self.current_file_running = {
            "location": location,
            "date": date,
            "time": time
        }
        file_address = (
            self.params.cache_dir + "/" +
            f"{self._get_filename(location, date, time)}_prepared_sq_tasks_"
            f"{self.geo_loader.get_hash_str()}_{self.params.get_hash_str(['cache_dir', 'free_flow_speed', 'dt', 'q_max'])}.json"
        )
        if os.path.isfile(file_address):
            self.tasks = json.load(open(file_address, "r", encoding="utf-8"))
            for index in range(len(self.tasks)):
                self.tasks[index]["dt"] = self.tasks[index]["dt"] * Units.S
                self.tasks[index]["q_max_up"] = self.tasks[index]["q_max_up"] * Units.PER_HR
                self.tasks[index]["q_max_down"] = self.tasks[index]["q_max_down"] * Units.PER_HR
                self.tasks[index]["k_j"] = self.tasks[index]["k_j"] * Units.PER_KM
                self.tasks[index]["link_length"] = self.tasks[index]["link_length"] * Units.M
            return
        tasks = []
        for link_id, cell_dict in self.cumulative_counts_dict.items(): # type: ignore
            for trajectory_time, data in cell_dict.items():
                tasks.append(
                    {
                        "q_max_up": self.params.q_max,
                        "q_max_down": self.params.q_max,
                        "next_occupancy": sum(self.next_timestamp_occupancy_dict[link_id][trajectory_time]["next_occupancy"]),
                        "cummulative_count_upstream_offset": data["cummulative_count_upstream_offset"],
                        "cummulative_count_upstream": data["cummulative_count_upstream"],
                        "cummulative_count_downstream": data["cummulative_count_downstream"],
                        "current_number_of_vehicles": data["current_number_of_vehicles"],
                        "inflow": self.first_cell_inflow_dict[link_id].get(trajectory_time, 0),
                        "dt": self.params.dt,
                        "trajectory_time": trajectory_time,
                        "tl_status": self.tl_status(trajectory_time, link_id),
                        "link_id": link_id,
                        "k_j": self.params.jam_density_link,
                        "entry_count": data["entry_count"],
                        "link_length": self.geo_loader.links[link_id].get_length(),
                        "next_exit": self.next_exit_link_values[link_id][trajectory_time]
                    }
                )
        self.tasks = tasks
        with open(file_address, "w", encoding="utf-8") as f:
            copy_tasks = deepcopy(tasks)
            for index in range(len(copy_tasks)):
                copy_tasks[index]["dt"] = copy_tasks[index]["dt"].to(Units.S).value
                copy_tasks[index]["q_max_up"] = copy_tasks[index]["q_max_up"].to(Units.PER_HR).value
                copy_tasks[index]["q_max_down"] = copy_tasks[index]["q_max_down"].to(Units.PER_HR).value
                copy_tasks[index]["k_j"] = copy_tasks[index]["k_j"].to(Units.PER_KM).value
                copy_tasks[index]["link_length"] = copy_tasks[index]["link_length"].to(Units.M).value
            json.dump(copy_tasks, f, indent=4)
        self.destruct()



    def prepare_ltm_tasks(self, location, date, time):
        """
        Prepares the necessary tasks for the specified location, date, and time.
        """
        self.activate_cummulative_counts_ltm(location, date, time)
        self.activate_tl_status_dict(location, date, time)
        self.activate_next_timestamp_occupancy(location, date, time)
        self.activate_next_exit_link(location, date, time)
        self.activate_first_cell_inflow_dict(location, date, time)
        self.current_file_running = {
            "location": location,
            "date": date,
            "time": time
        }
        file_address = (
            self.params.cache_dir + "/" +
            f"{self._get_filename(location, date, time)}_prepared_ltm_tasks_"
            f"{self.geo_loader.get_hash_str()}_{self.params.get_hash_str(['cache_dir', 'free_flow_speed', 'dt', 'wave_speed', 'jam_density_link'])}.json"
        )
        if os.path.isfile(file_address):
            self.tasks = json.load(open(file_address, "r", encoding="utf-8"))
            for index in range(len(self.tasks)):
                self.tasks[index]["dt"] = self.tasks[index]["dt"] * Units.S
                self.tasks[index]["wave_speed"] = self.tasks[index]["wave_speed"] * Units.KM_PER_HR
                self.tasks[index]["link_length"] = self.tasks[index]["link_length"] * Units.M
                self.tasks[index]["x"] = self.tasks[index]["x"] * Units.M
                self.tasks[index]["jam_density_link"] = self.tasks[index]["jam_density_link"] * Units.PER_KM
            return
        tasks = []
        for link_id, cell_dict in self.cumulative_counts_dict.items(): # type: ignore
            for trajectory_time, data in cell_dict.items():
                for cell_id, cell in self.geo_loader.links[link_id].cells.items():
                    if cell_id - 1 >= len(self.next_timestamp_occupancy_dict[link_id][trajectory_time]["next_occupancy"]):
                        continue

                    tasks.append(
                        {
                            "link_id": link_id,
                            "trajectory_time": trajectory_time,
                            "upstream_value_freeflow_with_eps_x": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["upstream_value_freeflow_with_eps_x"],
                            "downstream_value_freeflow_with_eps_x": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["downstream_value_freeflow_with_eps_x"],
                            "upstream_value_freeflow_with_eps_t": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["upstream_value_freeflow_with_eps_t"],
                            "downstream_value_freeflow_with_eps_t": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["downstream_value_freeflow_with_eps_t"],
                            "upstream_value_freeflow": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["upstream_value_freeflow"],
                            "downstream_value_freeflow": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["downstream_value_freeflow"],
                            "inflow": self.first_cell_inflow_dict[link_id].get(trajectory_time, 0),
                            "upstream_value_wavespeed_with_eps_x": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["upstream_value_wavespeed_with_eps_x"],
                            "downstream_value_wavespeed_with_eps_x": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["downstream_value_wavespeed_with_eps_x"],
                            "upstream_value_wavespeed_with_eps_t": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["upstream_value_wavespeed_with_eps_t"],
                            "downstream_value_wavespeed_with_eps_t": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["downstream_value_wavespeed_with_eps_t"],
                            "upstream_value_wavespeed": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["upstream_value_wavespeed"],
                            "downstream_value_wavespeed": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["downstream_value_wavespeed"],
                            "cell_id": cell_id,
                            "link_id": link_id,
                            "wave_speed": self.params.wave_speed,
                            "jam_density_link": self.params.jam_density_link,
                            "trajectory_time": trajectory_time,
                            "x": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["x"] * Units.M,
                            "link_length": self.cumulative_counts_dict[link_id][trajectory_time][cell_id]["link_length"] * Units.M,
                            "dt": self.params.dt,
                            "next_occupancy": self.next_timestamp_occupancy_dict[link_id][trajectory_time]["next_occupancy"][cell_id-1],
                            "next_exit": self.next_exit_link_values[link_id][trajectory_time]
                        }
                    )
        self.tasks = tasks
        with open(file_address, "w", encoding="utf-8") as f:
            copy_tasks = deepcopy(tasks)
            for index in range(len(copy_tasks)):
                copy_tasks[index]["dt"] = copy_tasks[index]["dt"].to(Units.S).value
                copy_tasks[index]["wave_speed"] = copy_tasks[index]["wave_speed"].to(Units.KM_PER_HR).value
                copy_tasks[index]["link_length"] = copy_tasks[index]["link_length"].to(Units.M).value
                copy_tasks[index]["x"] = copy_tasks[index]["x"].to(Units.M).value
                copy_tasks[index]["jam_density_link"] = copy_tasks[index]["jam_density_link"].to(Units.PER_KM).value

            json.dump(copy_tasks, f, indent=4)
        self.destruct()
        self.tasks = tasks
    
    def get_average_speeds_per_cell(self, location, date, time):
        """
        Returns a dictionary with dict[link_id][trajectory_time] = list(average speed)
        """
        # nbbi: Needs test
        file_address = (
            self.params.cache_dir + "/" +
            f"{self._get_filename(location, date, time)}_average_speeds_per_cell_" +
            f"{self.geo_loader.get_hash_str()}.json"
        )
        if os.path.isfile(file_address):
            with open(file_address, "r", encoding="utf-8") as f:
                average_speeds_per_cell = json.load(f)
            return convert_keys_to_float(average_speeds_per_cell)
        
        wlc_df = self.files_dict.get(
            (location, date, time), None
        )
        if wlc_df is None:
            raise ValueError(
                f"File not found for {location}, {date}, {time}, "
                "for processing average speeds"
            )
        wlc_df = pl.read_csv(wlc_df)
        wlc_df = wlc_df.with_columns(
            pl.col("trajectory_time").cast(pl.Float64).round(2)
        )
        groups = wlc_df.group_by(["link_id", "cell_id"])
        groups_length = wlc_df.select(["link_id", "cell_id"]).unique().height
        average_speeds = {}
        min_time = wlc_df["trajectory_time"].min()
        max_time = wlc_df["trajectory_time"].max()
        completed_df = pl.DataFrame({})
        for name, group in tqdm(groups, desc="Finding average speeds", total=groups_length):
            group = group.sort(["trajectory_time"])
            link_id, cell_id = name[0], name[1]
            filled_group = fill_missing_timestamps(
                group,
                "trajectory_time",
                self.time_interval,
                min_time, # type: ignore
                max_time # type: ignore
            )
            filled_group = filled_group.with_columns([
                pl.col("speed").cast(pl.Float64).forward_fill().backward_fill(),
                pl.col("link_id").fill_null(link_id),
                pl.col("cell_id").fill_null(cell_id),
                pl.col("trajectory_time").cast(pl.Float64).round(2)
            ])
            completed_df = pl.concat([completed_df, filled_group])
        
        groups = completed_df.group_by(["link_id", "trajectory_time"])
        groups_length = completed_df.select(["link_id", "trajectory_time"]).unique().height
        all_average_speeds = {}
        for name, group in tqdm(groups, desc="Finding average speeds", total=groups_length):
            link_id, trajectory_time = name[0], name[1]
            summation_speed = {cell_id: 0 for cell_id in self.geo_loader.links[link_id].cells.keys()}
            count = {cell_id: 0 for cell_id in self.geo_loader.links[link_id].cells.keys()}
            average_speed = {cell_id: 0.0 for cell_id in self.geo_loader.links[link_id].cells.keys()}
            for row in group.iter_rows(named=True):
                cell_id = row["cell_id"]
                speed = row["speed"]
                summation_speed[cell_id] += speed
                count[cell_id] += 1
            
            for cell_id in self.geo_loader.links[link_id].cells.keys():
                if count[cell_id] != 0:
                    average_speed[cell_id] = summation_speed[cell_id] / count[cell_id]
                else:
                    average_speed[cell_id] = 0.0
            if link_id not in all_average_speeds:
                all_average_speeds[link_id] = {}
            all_average_speeds[link_id][trajectory_time] = list(dict(sorted(average_speed.items(), key= lambda x: x[0])).values())

        with open(file_address, "w", encoding="utf-8") as f:
            json.dump(all_average_speeds, f, indent=4)
        return all_average_speeds
        
        

    def prepare_pw_tasks(self, location, date, time):
        """
        Preparing pw tasks
        """
        self.activate_occupancy_density_entry_exit_dict(location, date, time, "density")
        average_speeds = self.get_average_speeds_per_cell(location, date, time)
        self.activate_tl_status_dict(location, date, time)
        self.activate_next_timestamp_occupancy(location, date, time)
        self.activate_first_cell_inflow_dict(location, date, time)



        file_address = (
            self.params.cache_dir + "/" +
            f"{self._get_filename(location, date, time)}_prepared_pw_tasks_"
            f"{self.geo_loader.get_hash_str()}_{self.params.get_hash_str(['cache_dir', 'free_flow_speed', 'dt', 'jam_density_link'])}.json"
        )

        self.current_file_running = {
            "location": location,
            "date": date,
            "time": time
        }

        if not isinstance(self.cell_vector_occupancy_or_density_dict, dict):
            raise ValueError(f"self.cell_vector_occupancy_or_density_dict should be a dictionary. Got {type(self.cell_vector_occupancy_or_density_dict)}")

        if os.path.isfile(file_address):
            self.tasks = json.load(open(file_address, "r", encoding="utf-8"))
            for index in range(len(self.tasks)):
                self.tasks[index]["dt"] = self.tasks[index]["dt"] * Units.S
                self.tasks[index]["jam_density_link"] = self.tasks[index]["jam_density_link"] * Units.PER_KM
                self.tasks[index]["cell_lengths"] = [length * Units.M for length in self.tasks[index]["cell_lengths"]]
                self.tasks[index]["densities"] = [density * Units.PER_M for density in self.tasks[index]["densities"]]
                self.tasks[index]["speeds"] = [speed * Units.KM_PER_HR for speed in self.tasks[index]["speeds"]]
                self.tasks[index]["free_flow_speed"] = self.tasks[index]["free_flow_speed"] * Units.KM_PER_HR
            return
        
        self.tasks = []
        
        for link_id in tqdm(self.cell_vector_occupancy_or_density_dict.keys(), desc="Preparing pw tasks"):
            if link_id == 5:
                continue
            if not isinstance(self.cell_vector_occupancy_or_density_dict[link_id], dict):
                raise ValueError(f"self.cell_vector_occupancy_or_density_dict[{link_id}] should be a dictionary. Got {type(self.cell_vector_occupancy_or_density_dict[link_id])}")
            for trajectory_time in self.cell_vector_occupancy_or_density_dict[link_id].keys():
                
                density = self.cell_vector_occupancy_or_density_dict[link_id][trajectory_time]
                density_unit = [
                    (next_occupancy / self.geo_loader.links[link_id].cells[i+1].get_length()) 
                    for i, next_occupancy in enumerate(
                        self.next_timestamp_occupancy_dict[link_id][trajectory_time]["next_occupancy"]
                    )
                ]

                cell_length = [self.geo_loader.links[link_id].cells[i+1].get_length() for i in range(len(density))]
                speeds = average_speeds[link_id][trajectory_time]
                speeds_unit = [speed * Units.KM_PER_HR for speed in speeds]
                if len(density_unit) != len(cell_length):
                    raise ValueError(f"Density and cell length should be the same length. Got {len(density_unit)} and {len(cell_length)}")
                if len(density_unit) != len(speeds_unit):
                    raise ValueError(f"Density and speeds should be the same length. Got {len(density_unit)} and {len(speeds_unit)}")
                next_occupancy = self.next_timestamp_occupancy_dict[link_id][trajectory_time]["next_occupancy"]
                tl_status = self.tl_status(trajectory_time, link_id)
                self.tasks.append(
                    {
                        "link_id": link_id,
                        "trajectory_time": trajectory_time,
                        "densities": density_unit,
                        "cell_lengths": cell_length, # It's a list!
                        "speeds": speeds_unit,
                        "dt": self.params.dt,
                        "inflow": self.first_cell_inflow_dict[link_id].get(trajectory_time, 0),
                        "jam_density_link": self.params.jam_density_link,
                        "tl_status": tl_status,
                        "free_flow_speed": self.params.free_flow_speed,
                        "next_occupancy": next_occupancy,
                    }
                )
        with open(file_address, "w", encoding="utf-8") as f:
            copy_tasks = deepcopy(self.tasks)
            for index in range(len(copy_tasks)):
                copy_tasks[index]["dt"] = copy_tasks[index]["dt"].to(Units.S).value
                copy_tasks[index]["jam_density_link"] = copy_tasks[index]["jam_density_link"].to(Units.PER_KM).value
                copy_tasks[index]["cell_lengths"] = [length.to(Units.M).value for length in copy_tasks[index]["cell_lengths"]]
                copy_tasks[index]["densities"] = [density.to(Units.PER_M).value for density in copy_tasks[index]["densities"]]
                copy_tasks[index]["speeds"] = [speed.to(Units.KM_PER_HR).value for speed in copy_tasks[index]["speeds"]]
                copy_tasks[index]["free_flow_speed"] = copy_tasks[index]["free_flow_speed"].to(Units.KM_PER_HR).value
            json.dump(copy_tasks, f, indent=4)
        self.destruct()
        
    def prepare(self, class_name: str, fp_location: str, fp_date: str, fp_time: str):
        """
        Prepares the data for the specified class name.
        
        Args:
            class_name (str): The name of the class to prepare data for.
            fp_location (str): The location of the file.
            fp_date (str): The date of the file.
            fp_time (str): The time of the file.
        """
        if class_name == "CTM":
            self.prepare_ctm_tasks(
                fp_location,
                fp_date,
                fp_time
            )
        elif class_name == "PointQueue":
            self.prepare_pq_tasks(
                fp_location,
                fp_date,
                fp_time
            )
        elif class_name == "SpatialQueue":
            self.prepare_sq_tasks(
                fp_location,
                fp_date,
                fp_time
            )
        elif class_name == "LTM":
            self.prepare_ltm_tasks(
                fp_location,
                fp_date,
                fp_time
            )
        elif class_name == "PW":
            self.prepare_pw_tasks(
                fp_location,
                fp_date,
                fp_time
            )
        else:
            raise ValueError(f"Unknown class name: {class_name}")

    

if __name__ == "__main__":
    params = Parameters()
    intersection_locations = (
        pl.read_csv(".cache/traffic_lights.csv")
        .to_numpy()
        .tolist()   # It's format is [lat, lon]
    )
    intersection_locations = [
        POINT(loc[1], loc[0])
        for loc in intersection_locations
    ]  # It's format is [lat, lon]
    geo_loader = GeoLoader(
        locations=intersection_locations,
        cell_length=20
    )
    dl = DataLoader(
        fp_location="d1",
        fp_date="20181029",
        fp_time="0800_0830",
        geo_loader=geo_loader,
        params=params
    )
    