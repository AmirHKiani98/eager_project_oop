"""
A module for gererating test data for unit tests.
This uses sumo library to generate a random network and then generates random trips
for the network. (Traci_
"""
import traci
from pathlib import Path
TLS = [
    "cluster_300400470_5415104194_6175368841_6175368842",
    "GS_cluster_10134104320_250691755",
    "GS_250691739",
    "cluster_250691791_5697162223_6220792260_952119412"
]


class Generator:

    def __init__(self) -> None:
        self.base_dir = Path(__file__).parent.parent.parent
        self.net_file = self.base_dir / "tests" / "assets" / "sumo_net" / "osm.net.xml.gz"
        self.route_file = self.base_dir / "tests" / "assets" / "sumo_net" / "trips.trips.xml"
        self.sumo_config_file = self.base_dir / "tests" / "assets" / "sumo_net" / "osm.sumocfg"
    
    def start_sim(self):
        """
        Start the simulation using the sumo library.
        """
        traci.start(["sumo", "-c", str(self.sumo_config_file), "--step-length", "0.04"])    

        
    

if __name__ == "__main__":
    generator = Generator()
    generator.start_sim()

    # generator.generate_trips()
    # generator.generate_net()