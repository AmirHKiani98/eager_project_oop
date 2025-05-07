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
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
from more_itertools import chunked
from shapely.geometry import Point as POINT
import requests
from tqdm import tqdm
import polars as pl
from src.preprocessing.geo_loader import GeoLoader
from src.preprocessing.utility import fill_missing_timestamps

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
        cache_dir=".cache",
        line_threshold=20,
        time_interval=0.04
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
        self.base_url = "https://open-traffic.epfl.ch/wp-content/uploads/mydownloads.php"
        self.fp_location = [fp_location] if isinstance(fp_location, str) else fp_location
        self.fp_date = [fp_date] if isinstance(fp_date, str) else fp_date
        self.fp_time = [fp_time] if isinstance(fp_time, str) else fp_time
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.files_dict = {}
        self.density_files_dict = {}
        self.geo_loader = geo_loader
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
        return os.path.join(self.cache_dir, self._get_filename(location, date, time) + ".csv")

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
            print(f"File already exists: {self.get_cached_filepath(location, date, time)}")
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

            print(f"Downloaded to: {self.get_cached_filepath(location, date, time)}")
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
                        exploded_file_address = self._explode_dataset(
                            raw_data_file_path, location, date, time
                        )
                        processed_file_address = self._process_link_cell(
                            exploded_file_address, location, date, time
                        )
                        vehicle_on_corridor_address = self._get_vehicle_on_corridor_df(
                            processed_file_address, location, date, time
                        )
                        removed_vehicles_on_minor_roads = self._remove_vehicle_on_minor_roads(
                            vehicle_on_corridor_address, location, date, time
                        )

                        self.files_dict[
                            (location, date, time)
                        ] = removed_vehicles_on_minor_roads

                        self.density_files_dict[
                            (location, date, time)
                        ] = self._get_density_df(
                            removed_vehicles_on_minor_roads, location, date, time
                        )

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
            # return pl.DataFrame({})  # Skip bad line

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
                data["trajectory_time"].append(item)
                data["track_id"].append(track_id)
                data["veh_type"].append(veh_type)
                data["traveled_d"].append(traveled_d)
                data["avg_speed"].append(avg_speed)
        return pl.DataFrame(data)


    def _explode_dataset(self, raw_data_location, location, date, time):
        file_address = (
            self.cache_dir + "/" + self._get_filename(location, date, time) + "_exploded.csv"
        )
        if os.path.isfile(file_address):
            return file_address
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
            self.cache_dir + "/" + self._get_filename(location, date, time) + "_withlinkcell_" +
            self.geo_loader.get_hash_str() + ".csv"
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
            link_ids.append(link.link_id)
            link_distances.append(link_distance)
            cell_ids.append(cell.cell_id)
            cell_distances.append(cell_distance)

        # Add columns to the DataFrame
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
            self.cache_dir + "/" + self._get_filename(location, date, time) +
            "_vehicle_on_corridor_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address
        # Loading the dataframe with link and cell (wlc_df)
        wlc_df = pl.read_csv(with_link_cell_address)
        wlc_df = wlc_df.filter(pl.col("distance_from_link") < self.line_threshold)
        wlc_df.write_csv(file_address)
        return file_address

    def _remove_vehicle_on_minor_roads(self, vehicle_on_corridor_address,  location, date, time):
        """
        Removes vehicles that are on minor roads from the DataFrame.
        """
        file_address = (
            self.cache_dir + "/" + self._get_filename(location, date, time) +
            "_vehicle_on_minor_roads_removed_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address

        wlc_df = pl.read_csv(vehicle_on_corridor_address)
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

        wlc_df = wlc_df.filter(~pl.col("track_id").is_in(removed_ids))
        wlc_df.write_csv(file_address)
        return file_address

    def _get_density_df(self, fully_processed_file_address, location, date, time):
        """
        The fully addressed file is the one that has been exploded, processed, and filtered
        which refers to the file address _remove_vehicle_on_minor_roads returned.
        """
        file_address = (
            self.cache_dir + "/" + self._get_filename(location, date, time)
            + "_density_" + self.geo_loader.get_hash_str() + ".parquet"
        )

        if os.path.isfile(file_address):
            return file_address

        wlc_df = pl.read_csv(fully_processed_file_address)
        min_time = wlc_df["trajectory_time"].min()
        max_time = wlc_df["trajectory_time"].max()
        counts = wlc_df.group_by(["link_id", "cell_id", "trajectory_time"]).agg([
            pl.col("track_id").alias("vehicle_ids")
        ])
        counts = counts.sort(["link_id", "cell_id", "trajectory_time"])
        complete_counts = pl.DataFrame({})
        groups = counts.group_by(["link_id", "cell_id"])
        num_groups = wlc_df.select(["link_id", "cell_id"]).unique().height
        for _, group in tqdm(groups, total=num_groups, desc="Counting vehicles"):
            group = fill_missing_timestamps(group, min_time, max_time, self.time_interval)
            group = group.with_columns(
                pl.col("vehicle_ids").fill_null([])  # sets default to empty list
            )
            group = group.with_columns([
                pl.col("vehicle_ids").shift(1).alias("prev_vehicles"),
                pl.col("vehicle_ids").shift(-1).alias("next_vehicles")
            ])

            group = group.with_columns([
                # Vehicles that appeared now but weren’t there before = entries
                pl.struct(["vehicle_ids", "prev_vehicles"])
                .map_elements(lambda s: list(set(s["vehicle_ids"]) - set(s["prev_vehicles"] or [])),
                                return_dtype=pl.List(pl.Int64))
                .alias("entries"),

                # Vehicles that were there but not anymore = exits
                pl.struct(["vehicle_ids", "next_vehicles"])
                .map_elements(lambda s: list(set(s["vehicle_ids"]) - set(s["next_vehicles"] or [])),
                              return_dtype=pl.List(pl.Int64))
                .alias("exits")
            ])
            group = group.drop(["prev_vehicles", "next_vehicles"])
            complete_counts = pl.concat([complete_counts, group])
        complete_counts = complete_counts.with_columns([
            pl.col("entries").list.len().alias("entry_count"),
            pl.col("exits").list.len().alias("exit_count")
        ])
        complete_counts.write_parquet(file_address)
        print(f"Density DataFrame saved to {file_address}")
        return file_address

    def _get_traffic_light_status(self, fully_processed_file_address, location, date, time):
        """
        Returns the traffic light status for the specified location, date, and time.
        """
        file_address = (
            self.cache_dir + "/" + self._get_filename(location, date, time) +
            "_traffic_light_status_" + self.geo_loader.get_hash_str() + ".csv"
        )
        if os.path.isfile(file_address):
            return file_address

        # Traffic light locations
        traffic_light_locations = self.geo_loader.get_locations()
        wlc_df = pl.read_csv(fully_processed_file_address)

        return file_address


# Run as script
if __name__ == "__main__":
    # Example usage
    intersection_locations = pl.read_csv(".cache/traffic_lights.csv").to_numpy().tolist()
    intersection_locations = [POINT(loc[1], loc[0]) for loc in intersection_locations]
    model_geo_loader = GeoLoader(
        locations=intersection_locations,
        cell_length=20.0
        )
    dl = DataLoader(
        fp_location=["d1"],
        fp_date=["20181029"],
        fp_time=["0800_0830", "0830_0900", "0900_0930", "0930_1000", "1000_1030"],
        geo_loader=model_geo_loader
    )
