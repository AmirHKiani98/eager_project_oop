import os
import requests
from tqdm import tqdm

class DataLoader:
    def __init__(self, fp_location: str, fp_date: str, fp_time: str, cache_dir="downloads"):
        self.base_url = "https://open-traffic.epfl.ch/wp-content/uploads/mydownloads.php"
        self.fp_location = fp_location
        self.fp_date = fp_date
        self.fp_time = fp_time
        self.cache_dir = cache_dir
        self.validate_inputs()
        self.payload = self._build_payload()
        os.makedirs(self.cache_dir, exist_ok=True)

    def handle_data(self):
        """
        
        """
    def validate_inputs(self):
        if not (self.fp_location.startswith("d") and (self.fp_location[1:].isdigit() or self.fp_location == "dX")):
            raise ValueError(f"Invalid fp_location: {self.fp_location}")
        
        if len(self.fp_date) != 8 or not self.fp_date.isdigit():
            raise ValueError(f"Invalid fp_date format: {self.fp_date}")
        
        if not (self.fp_time.endswith("00") or self.fp_time.endswith("30")):
            raise ValueError(f"Invalid fp_time format: {self.fp_time}")

    def _build_payload(self) -> str:
        return f"fpLocation={self.fp_location}&fpDate={self.fp_date}&fpTime={self.fp_time}"

    def get_download_url(self) -> str:
        return f"{self.base_url}?{self.payload}"

    def get_filename(self) -> str:
        return f"{self.fp_location}_{self.fp_date}_{self.fp_time}.csv"

    def get_cached_filepath(self) -> str:
        return os.path.join(self.cache_dir, self.get_filename())

    def check_file_exists_in_cache(self) -> bool:
        return os.path.isfile(self.get_cached_filepath())

    def download_file(self) -> str:
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
