"""
This is the main entry point for the application.
# It initializes the necessary components and starts the traffic simulation.
"""
import argparse
import logging
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
from src.common_utility.units import Units
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
      `.cache/traffic_lights.csv`.
    - The POINT objects are created using the latitude and longitude values from
      the CSV file.
    """
    logging.basicConfig(level=logging.INFO)

    params = Parameters(
        vehicle_length=5 * Units.M,
        free_flow_speed=50 * Units.KM_PER_HR,
        wave_speed=10 * Units.KM_PER_HR,
        num_lanes=3,
        jam_density_link=150 * Units.PER_KM,
        dt=1 * Units.S,
        q_max=2500 * Units.PER_HR,
    )

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
        default=".cache/traffic_lights.csv"
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run calibration for the model"
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
    model_geo_loader = GeoLoader(
        locations=intersection_locations,
        cell_length=20.0
        )
    dl = DataLoader(
        params=params,
        fp_location=args.fp_location,
        fp_date=args.fp_date,
        fp_time=args.fp_time,
        geo_loader=model_geo_loader
    )
    batch_size = args.batch_size
    if args.model == "ctm":
        model = CTM(
            dl=dl,
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        num_processes = cpu_count()
        if not args.calibration:
            model.run_with_multiprocessing(num_processes=num_processes, batch_size=batch_size)
        else:
            dl.prepare("CTM", args.fp_location, args.fp_date, args.fp_time) # This will be automatically handled in run_with_multiprocessing
            model.run_calibration(num_processes=num_processes, batch_size=batch_size)
    elif args.model == "pq":
        model = PointQueue(
            dl=dl,
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        num_processes = cpu_count()
        if not args.calibration:
            model.run_with_multiprocessing(num_processes=num_processes, batch_size=batch_size)
        else:
            dl.prepare("PointQueue", args.fp_location, args.fp_date, args.fp_time)
            model.run_calibration(num_processes=num_processes, batch_size=batch_size)
    elif args.model == "sq":
        model = SpatialQueue(
            dl=dl,
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        num_processes = cpu_count()
        if not args.calibration:
            model.run_with_multiprocessing(num_processes=num_processes, batch_size=batch_size)
        else:
            dl.prepare("SpatialQueue", args.fp_location, args.fp_date, args.fp_time)
            model.run_calibration(num_processes=num_processes, batch_size=batch_size)
    elif args.model == "ltm":
        model = LTM(
            dl=dl,
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        num_processes = cpu_count()
        if not args.calibration:
            model.run_with_multiprocessing(num_processes=num_processes, batch_size=batch_size)
        else:
            dl.prepare("LTM", args.fp_location, args.fp_date, args.fp_time)
            model.run_calibration(num_processes=num_processes, batch_size=batch_size)
    elif args.model == "pw":
        model = PW(
            dl=dl,
            fp_location=args.fp_location,
            fp_date=args.fp_date,
            fp_time=args.fp_time,
        )
        num_processes = cpu_count()
        if not args.calibration:
            model.run_with_multiprocessing(num_processes=num_processes, batch_size=batch_size)
        else:
            dl.prepare("PW", args.fp_location, args.fp_date, args.fp_time)
            model.run_calibration(num_processes=num_processes, batch_size=batch_size)
    else:
        raise ValueError(f"Model {args.model} not supported")

if __name__ == "__main__":
    main()
