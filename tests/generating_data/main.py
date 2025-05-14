"""
Main module for running the generator.
This module is used to generate test data for the unit tests.
"""
import argparse
from tests.generating_data.generator import Generator

def main():
    """
    Main function for running the generator.
    """
    parser = argparse.ArgumentParser(description="Generate test data for unit tests.")
    parser.add_argument(
        "--simulation-time",
        type=int,
        default=800,
        help="Simulation time in seconds. Default is 800 seconds.",
    )
    parser.add_argument(
        "--cell-length",
        type=float,
        default=None,
        help="Length of each cell in meters. Default is None.",
    )
    parser.add_argument(
        "--cell-numbers",
        type=int,
        default=None,
        help="Number of cells in the lane. Default is None.",
    )
    args = parser.parse_args()
    # Check if cell_length and cell_numbers are provided
    if args.cell_length is None and args.cell_numbers is None:
        raise ValueError("You must provide either cell_length or cell_numbers to the generator.")
    generator = Generator(simulation_time=args.simulation_time, cell_length=args.cell_length, cell_numbers=args.cell_numbers)
    generator.run()

if __name__ == "__main__":
    main()