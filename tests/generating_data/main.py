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
    args = parser.parse_args()

    generator = Generator(simulation_time=args.simulation_time)
    generator.run()

if __name__ == "__main__":
    main()