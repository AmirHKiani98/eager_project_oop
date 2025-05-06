import os
import requests
from tqdm import tqdm

class DataLoader:
    def __init__(self, fp_location: str, fp_date: str, fp_time: str, cache_dir=".cache"):
        
        """
        Initializes the DataLoader with the specified parameters.

        Args:
            fp_location (str): Specifies the location identifier. 
                - Can be 'd1' to 'd10' for specific regions or 'dX' for all 10 regions.
            fp_date (str): Specifies the date in the format 'yyyymmdd'. 
                - Valid dates include '20181024', '20181029', '20181031', or '20181101'.
            fp_time (str): Specifies the time range in the format 'hhmm_hhmm'. 
                - The time range should have a 30-minute difference between start and end times.
            cache_dir (str, optional): Directory where downloaded files will be cached. 
                - Defaults to "downloads".

        Attributes:
            base_url (str): The base URL for downloading data.
            fp_location (str): The location identifier provided during initialization.
            fp_date (str): The date provided during initialization.
            fp_time (str): The time range provided during initialization.
            cache_dir (str): The directory for caching downloaded files.
            payload (dict): The payload built from the provided inputs for making requests.

        Raises:
            ValueError: If any of the input parameters are invalid.

        Notes:
            - The `validate_inputs` method is called to ensure the inputs are valid.
            - The `cache_dir` directory is created if it does not already exist.
        
        """
        self.base_url = "https://open-traffic.epfl.ch/wp-content/uploads/mydownloads.php"
        self.fp_location = fp_location
        self.fp_date = fp_date
        self.fp_time = fp_time
        self.cache_dir = cache_dir
        self.validate_inputs()
        self.payload = self._build_payload()
        os.makedirs(self.cache_dir, exist_ok=True)

    
        
    def validate_inputs(self):
        """
        Validates the input attributes of the class instance.
        This method checks the following conditions:
        1. `fp_location` must start with the letter 'd' and be followed by either:
           - A sequence of digits, or
           - The string "dX".
        2. `fp_date` must be exactly 8 characters long and consist only of digits.
        3. `fp_time` must end with either "00" or "30".
        Raises:
            ValueError: If any of the above conditions are not met, a ValueError is raised
                        with a message indicating the specific invalid attribute.
        """
        if not (self.fp_location.startswith("d") and (self.fp_location[1:].isdigit() or self.fp_location == "dX")):
            raise ValueError(f"Invalid fp_location: {self.fp_location}")
        
        if len(self.fp_date) != 8 or not self.fp_date.isdigit():
            raise ValueError(f"Invalid fp_date format: {self.fp_date}")
        
        if not (self.fp_time.endswith("00") or self.fp_time.endswith("30")):
            raise ValueError(f"Invalid fp_time format: {self.fp_time}")

    def _build_payload(self) -> str:
        """
        Constructs a payload string with file processing location, date, and time.

        Returns:
            str: A formatted string containing the file processing location (`fpLocation`),
                 date (`fpDate`), and time (`fpTime`) as key-value pairs.
        """
        return f"fpLocation={self.fp_location}&fpDate={self.fp_date}&fpTime={self.fp_time}"

    def get_download_url(self) -> str:
        """
        Constructs and returns the full download URL by combining the base URL 
        and the payload as query parameters.

        Returns:
            str: The complete download URL.
        """
        return f"{self.base_url}?{self.payload}"

    def get_filename(self) -> str:
        """
        Constructs and returns the filename based on the file location, date, and time.

        Returns:
            str: The constructed filename in the format "{fp_location}_{fp_date}_{fp_time}.csv".
        """
        return f"{self.fp_location}_{self.fp_date}_{self.fp_time}.csv"

    def get_cached_filepath(self) -> str:
        """
        Constructs and returns the full file path for a cached file.

        This method combines the cache directory path and the filename
        to generate the full path to the cached file.

        Returns:
            str: The full file path of the cached file.
        """
        return os.path.join(self.cache_dir, self.get_filename())

    def check_file_exists_in_cache(self) -> bool:
        """
        Check if the cached file exists in the specified location.

        This method determines whether a file exists at the path returned by 
        the `get_cached_filepath` method. It uses `os.path.isfile` to verify 
        the existence of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.isfile(self.get_cached_filepath())

    def download_file(self) -> str:
        """
        Downloads a file from a specified URL and saves it to a local cache.

        If the file already exists in the cache, it returns the cached file path.
        Otherwise, it sends a POST request to the specified URL with the provided
        headers and payload, streams the file content, and writes it to the cache
        while displaying a progress bar.

        Returns:
            str: The file path where the downloaded file is saved.

        Raises:
            RuntimeError: If the file download fails due to a non-200 HTTP response.

        Notes:
            - The method uses a progress bar to indicate the download progress.
            - The headers include a User-Agent and other metadata required for the request.
            - The file is downloaded in chunks to handle large files efficiently.
        """
        if self.check_file_exists_in_cache():
            print(f"File already exists: {self.get_cached_filepath()}")
            return self.get_cached_filepath()

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
            data=self.payload,
            stream=True
        )

        if response.status_code == 200:
            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 8192

            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=self.get_filename())

            with open(self.get_cached_filepath(), "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()
            print(f"Downloaded to: {self.get_cached_filepath()}")
            return self.get_cached_filepath()
        else:
            raise RuntimeError(f"Failed to download file: {response.status_code} - {response.text}")

# Run as script
if __name__ == "__main__":
    dl = DataLoader(fp_location="d1", fp_date="20181029", fp_time="0800_0830")
    dl.download_file()
