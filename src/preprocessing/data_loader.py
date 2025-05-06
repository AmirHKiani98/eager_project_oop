import os
import requests
from tqdm import tqdm

class DataLoader:
    def __init__(self, fp_location: str | list, fp_date: str | list, fp_time: str | list, cache_dir=".cache"):
        """
        Initializes the DataLoader with the specified parameters.

        Args:
            fp_location (str | list): Specifies the location identifier(s).
            fp_date (str | list): Specifies the date(s) in the format 'yyyymmdd'.
            fp_time (str | list): Specifies the time range(s) in the format 'hhmm_hhmm'.
            cache_dir (str, optional): Directory where downloaded files will be cached. Defaults to ".cache".
        """
        self.base_url = "https://open-traffic.epfl.ch/wp-content/uploads/mydownloads.php"
        self.fp_location = [fp_location] if isinstance(fp_location, str) else fp_location
        self.fp_date = [fp_date] if isinstance(fp_date, str) else fp_date
        self.fp_time = [fp_time] if isinstance(fp_time, str) else fp_time
        self.cache_dir = cache_dir
        self.validate_inputs()
        os.makedirs(self.cache_dir, exist_ok=True)

    def validate_inputs(self):
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

    def get_download_url(self, location, date, time) -> str:
        """
        Constructs and returns the full download URL.
        """
        return f"{self.base_url}?{self._build_payload(location, date, time)}"

    def get_filename(self, location, date, time) -> str:
        """
        Constructs and returns the filename.
        """
        return f"{location}_{date}_{time}.csv"

    def get_cached_filepath(self, location, date, time) -> str:
        """
        Constructs and returns the full file path for a cached file.
        """
        return os.path.join(self.cache_dir, self.get_filename(location, date, time))

    def check_file_exists_in_cache(self, location, date, time) -> bool:
        """
        Check if the cached file exists in the specified location.
        """
        return os.path.isfile(self.get_cached_filepath(location, date, time))

    def check_file_is_correct(self, downloaded_loc):
        """
        It checks if the file is not currupted.
        """
        with open(downloaded_loc, "w") as f:
            first_line = f.readline()
            if first_line == "Empty Dataset":
                os.remove(downloaded_loc)
                return False
            else:
                return True

    def download_file(self, location, date, time) -> str:
        """
        Downloads a file from a specified URL and saves it to a local cache.
        """
        if self.check_file_exists_in_cache(location, date, time):
            print(f"File already exists: {self.get_cached_filepath(location, date, time)}")
            return self.get_cached_filepath(location, date, time)

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Referer": "https://open-traffic.epfl.ch/index.php/downloads/",
            "Origin": "https://open-traffic.epfl.ch",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "*/*"
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            data=self._build_payload(location, date, time),
            stream=True
        )

        if response.status_code == 200:
            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 8192

            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=self.get_filename(location, date, time),
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

    def download_all_files(self):
        """
        Downloads all files for the specified combinations of location, date, and time.
        """
        for location in self.fp_location:
            for date in self.fp_date:
                for time in self.fp_time:
                    self.download_file(location, date, time)


# Run as script
if __name__ == "__main__":
    dl = DataLoader(fp_location=["d1", "d2"], fp_date=["20181029"], fp_time=["0800_0830", "0830_0900"])
    dl.download_all_files()
