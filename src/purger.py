"""
This module provides functionality for purging the cache.

It includes methods and utilities to clear cached data, ensuring that
stale or outdated information is removed and fresh data can be retrieved
or generated as needed.

Use this module to manage cache lifecycle and maintain optimal
application performance.
"""
import argparse
import os
from glob import glob
from tqdm import tqdm
class CachePurger:
    """
    A class to handle cache purging operations.

    Attributes:
        cache_directory (str): The directory where the cache is stored.
    """

    def __init__(
        self,
        cache_directory: str,
        specific_date: str = "",
        specific_time: str = "",
        specific_location: str = "",

    ):
        """
        Initializes the CachePurger with the specified cache directory.

        Args:
            cache_directory (str): The directory where the cache is stored.
        """
        self.cache_directory = cache_directory
        self.specific_date = specific_date
        self.specific_time = specific_time
        self.specific_location = specific_location
        self.specific_file = "_".join(
            [
                self.specific_location,
                self.specific_date,
                self.specific_time,
            ]
        )
        if self.specific_file.replace("_", "") == "":
            self.specific_file = ""

        print(self.specific_file)

    def purge_test_df_file(self):
        """
        Purges the cache by removing all files in the cache directory.

        Raises:
            Exception: If an error occurs while purging the cache.
        """
        pattern = os.path.join(self.cache_directory, f"*{self.specific_file}_test_df_*")
        files = glob(pattern)
        for _file in tqdm(files, desc="Purging test cache", unit="file"):
            try:
                os.remove(_file)
                print(f"Removed file: {_file}")
            except OSError as e:
                raise OSError(f"Error purging cache: {e}") from e

    def purge_all(self):
        """
        Purges the cache by removing all files in the cache directory.

        Raises:
            Exception: If an error occurs while purging the cache.
        """
        pattern = os.path.join(self.cache_directory, f"*{self.specific_file}*")
        files = glob(pattern)
        for _file in tqdm(files, desc="Purging all cache", unit="file"):
            try:
                os.remove(_file)
                print(f"Removed file: {_file}")
            except OSError as e:
                raise OSError(f"Error purging cache: {e}") from e

    def purge_traffic_light_files(self):
        """
        Purges the cache by removing all traffic light files in the cache directory.

        Raises:
            Exception: If an error occurs while purging the cache.
        """
        pattern = os.path.join(
            self.cache_directory,
            f"*{self.specific_file}*_traffic_light_*"
        )
        files = glob(pattern)
        for _file in tqdm(files, desc="Purging processed traffic light cache", unit="file"):
            try:
                os.remove(_file)
                print(f"Removed file: {_file}")
            except OSError as e:
                raise OSError(f"Error purging cache: {e}") from e
        print("Completed purging traffic light files.")
    
    def purge_density_entry_exit_files(self):
        """
        Purges the cache by removing all density entry and exit files in the cache directory.

        Raises:
            Exception: If an error occurs while purging the cache.
        """
        pattern = os.path.join(
            self.cache_directory,
            f"*{self.specific_file}*_density_entry_exit_*"
        )
        files = glob(pattern)
        for _file in tqdm(files, desc="Purging processed density entry/exit cache", unit="file"):
            try:
                os.remove(_file)
                print(f"Removed file: {_file}")
            except OSError as e:
                raise OSError(f"Error purging cache: {e}") from e
        print("Completed purging density entry/exit files.")

def main():
    """
    Parses command-line arguments and purges cached files based on the specified mode.

    This function sets up an argument parser to accept options for cache directory,
    location, date, time, and purge mode. Depending on the selected mode
    ('all', 'test', or 'traffic'), it invokes the corresponding purge method
    on a CachePurger instance.

    Args:
        None. Arguments are parsed from the command line.

    Raises:
        SystemExit: If argument parsing fails or if an invalid mode is provided.
    """

    parser = argparse.ArgumentParser(description="Purge cached files.")
    parser.add_argument("--cache-dir", default=".cache", help="Path to cache directory")
    parser.add_argument("--location", default="", help="Specific location")
    parser.add_argument("--date", default="", help="Specific date (yyyymmdd)")
    parser.add_argument("--time", default="", help="Specific time (hhmm_hhmm)")
    parser.add_argument("--mode", choices=["all", "test", "traffic", "density_entry_exit"], default="test")

    args = parser.parse_args()
    purger = CachePurger(args.cache_dir, args.date, args.time, args.location)

    if args.mode == "all":
        purger.purge_all()
    elif args.mode == "test":
        purger.purge_test_df_file()
    elif args.mode == "traffic":
        purger.purge_traffic_light_files()
    elif args.mode == "density_entry_exit":
        purger.purge_density_entry_exit_files()

if __name__ == "__main__":
    main()
