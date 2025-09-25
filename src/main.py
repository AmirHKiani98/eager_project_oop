"""
This is the main entry point for the application.
# It initializes the necessary components and starts the traffic simulation.
"""
import argparse
import logging
import os
from multiprocessing import cpu_count
from shapely.geometry import Point as POINT
import polars as pl

from src.model.ctm import CTM
from src.model.point_queue import PointQueue
from src.model.spatial_queue import SpatialQueue
from src.model.ltm import LTM
from src.model.pw import PW
from src.model.params import Parameters
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.geo_loader import GeoLoader
from src.visualization.plotter import Plotter

# from src.visualization.plotter import Plotter

def main():
    """
    Main function to initialize and configure the traffic simulation.

    This function sets up logging, loads intersection locations from a CSV file,
    initializes the GeoLoader and DataLoader with the required parameters, and
    defines simulation parameters using the Parameters class. It also initializes
    an argument parser for handling command-line arguments.

    Steps:
    1. Reads intersection locations from a CSV file and converts them into a list
        of POINT objects.
    2. Initializes a GeoLoader instance with the intersection locations and a
        specified cell length.
    3. Configures a DataLoader instance with file paths, dates, times, and the
        GeoLoader instance.
    4. Sets up simulation parameters such as vehicle length, free flow speed, wave
        speed, number of lanes, and jam density per link.
    5. Prepares an argument parser for handling command-line arguments.

    Note:
    - The CSV file containing intersection locations is expected to be located at
      `traffic_lights.csv`.
    - The POINT objects are created using the latitude and longitude values from
      the CSV file.
    """
    logging.basicConfig(level=logging.INFO)

    

    # Initialize argparse
    parser = argparse.ArgumentParser(description="Traffic Simulation")
    parser.add_argument(
        "--model",
        type=str,
        default="ctm"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000
    )
    parser.add_argument(
        "--fp-location",
        type=str,
        default="d1"
    )
    parser.add_argument(
        "--fp-date",
        type=str,
        default="20181029"
    )
    parser.add_argument(
        "--fp-time",
        type=str,
        default="0800_0830"
    )
    parser.add_argument(
        "--fp-geo",
        type=str,
        default="traffic_lights.csv"
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run calibration for the model"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory to store cache files"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=10.0,
        help="Time step for the simulation in seconds"
    )
    
    args = parser.parse_args()
    # Example usage

    
    intersection_locations = (
        pl.read_csv(args.fp_geo)
        .to_numpy()
        .tolist()
    )
    intersection_locations = [
        POINT(loc[1], loc[0])
        for loc in intersection_locations
    ]
    
    
    batch_size = args.batch_size
    if len(args.model.split(",")) > 1:
        for model_name in args.model.split(","):
            os.system(f"python -m src.main --model {model_name} --fp-location {args.fp_location} --fp-date {args.fp_date} --fp-time {args.fp_time} --batch-size {batch_size} --calibration")
    if args.model == "ctm":
        model = CTM(
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        
    elif args.model == "pq":
        model = PointQueue(
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        
    elif args.model == "sq":
        model = SpatialQueue(
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
    elif args.model == "ltm":
        model = LTM(
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        
    elif args.model == "pw":
        model = PW(
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
    elif args.model == "actual":
        params = Parameters(
            cache_dir=args.cache_dir
        )
        model = DataLoader(
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
            geo_loader=GeoLoader(intersection_locations, cell_length=5),
            params=params
        )
        model.prepare_actuals(
            location=args.fp_location,
            date=args.fp_date,
            time=args.fp_time
        )
        plotter = Plotter(cache_dir=args.cache_dir)
        plotter.plot_actuals(
            model.tasks
        )
        return
    

    else:
        raise ValueError(f"Model {args.model} not supported")
    num_processes = cpu_count()
    if not args.calibration:
        model.run_with_multiprocessing(num_processes=num_processes, batch_size=batch_size)
    else:
        # cache_dir, num_processes=None, batch_size=None, vehicle_length=5, num_lanes=1, dt=5, locations=None)
        model.run_calibration(cache_dir=args.cache_dir, locations=intersection_locations, num_processes=num_processes, batch_size=batch_size, vehicle_length=5, num_lanes=3, dt=args.dt)
if __name__ == "__main__":
    main()
