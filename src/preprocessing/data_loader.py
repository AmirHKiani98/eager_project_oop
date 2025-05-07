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
from pathlib import Path
from shapely.geometry import Point as POINT
import requests
from tqdm import tqdm
import polars as pl
from src.preprocessing.geo_loader import GeoLoader
from multiprocessing import Pool, cpu_count

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
        self.base_url = "https://open-traffic.epfl.ch/wp-content/uploads/mydownloads.php"
        self.fp_location = [fp_location] if isinstance(fp_location, str) else fp_location
        self.fp_date = [fp_date] if isinstance(fp_date, str) else fp_date
        self.fp_time = [fp_time] if isinstance(fp_time, str) else fp_time
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.files_list = {}
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
        return f"{location}_{date}_{time}.csv"

    def get_cached_filepath(self, location, date, time) -> str:
        """
        Constructs and returns the full file path for a cached file.
        """
        return os.path.join(self.cache_dir, self._get_filename(location, date, time))

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
                        exploded_file_address = self._explode_dataset(raw_data_file_path)
                        self.files_list[
                            (location, date, time)
                        ] = exploded_file_address
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


    def _explode_dataset(self, raw_data_location):
        file_address = '.cache/' + Path(raw_data_location).stem + "_exploded.csv"
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

        This method iterates over a list of file metadata (`self.files_list`),
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
        for (_, _, _), file_address in self.files_list.items():
            print("Address is", file_address)
            read_csv = pl.read_csv(file_address)
            dataframes.append(read_csv)
        if dataframes:
            self.df = pl.concat(dataframes)
        return self.df

    def find_links(self):
        """
        Finds the links in the DataFrame and assigns them to the GeoLoader.
        """
        if self.df.is_empty():
            raise ValueError("DataFrame is empty. Cannot find links.")
        # # Form the list of points from the DataFrame
        temp_df = self.df[500:800]
        points = [
            POINT(row["lon"], row["lat"])
            for row in tqdm(temp_df.iter_rows(named=True))
        ]
        # Find the closest link for each point
        with Pool(cpu_count()) as pool:
            closest_links = list(
                tqdm(
                    pool.imap(self.geo_loader.find_closest_link, points),
                    total=len(points),
                    desc="Finding closest links",
                    dynamic_ncols=True
                )
            )
        # Assign the closest links to the df
        # Extract link IDs and distances
        link_ids = [link.link_id for link, _ in closest_links]
        distances = [distance for _, distance in closest_links]

        # Add columns to the DataFrame
        temp_df = temp_df.with_columns([
            pl.Series("link_id", link_ids),
            pl.Series("distance_from_link", distances)
        ])
        return temp_df


# Run as script
if __name__ == "__main__":
    # Example usage
    intersection_locations = pl.read_csv(".cache/traffic_lights.csv").to_numpy().tolist()
    intersection_locations = [POINT(loc[1], loc[0]) for loc in intersection_locations]
    model_geo_loader = GeoLoader(
        locations=intersection_locations,
        cell_length=20.0,
        number_of_cells=2
    )
    dl = DataLoader(
        fp_location=["d1"],
        fp_date=["20181029"],
        fp_time=["0800_0830"],
        geo_loader=model_geo_loader
    )
    print(dl.find_links())
    # for row in dl.df.iter_rows(named=True):
    #     point = POINT(row["lon"], row["lat"])
    #     print(model_geo_loader.find_closest_link(point))
