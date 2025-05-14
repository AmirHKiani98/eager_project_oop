"""
A module for gererating test data for unit tests.
This uses sumo library to generate a random network and then generates random trips
for the network. (Traci_
"""
import traci
from pathlib import Path
from tests.generating_data.vehicle import Vehicle
TLS = [
    "cluster_300400470_5415104194_6175368841_6175368842",
    "GS_cluster_10134104320_250691755",
    "GS_250691739",
    "cluster_250691791_5697162223_6220792260_952119412"
]

MAJOR_EDGES = [
    "316770031#0",
    "168914458#0",
    "299510529#0",
    "299510529#1",
    "299644700#1",
    "299644700#2",
    "299644700#5",
    "299644700#6",
    "299644700#7",
    "299644700#8"
]


MINOR_EDGES = [
    "309269227",
    "299510530#0",
    "299510180#0",
    "207979566#5",
    "23182796#0",
    "23182843#0",
    "393398564#10",
    "295081893#2",
    "1052364155#2",
    "23182833#2",
    "1052364156#2",
    "23182837#1"
]

class Generator:

    def __init__(self) -> None:
        self.base_dir = Path(__file__).parent.parent.parent
        self.net_file = self.base_dir / "tests" / "assets" / "sumo_net" / "osm.net.xml"
        self.route_file = self.base_dir / "tests" / "assets" / "sumo_net" / "trips.trips.xml"
        self.sumo_config_file = self.base_dir / "tests" / "assets" / "sumo_net" / "osm.sumocfg"
    
    
    def start_sim(self):
        """
        Start the simulation using the sumo library.
        """
        traci.start(["sumo", "-n", str(self.net_file), "-r", str(self.route_file), "--step-length", "0.04"])    

    def run(self):
        """
        Step through the simulation.
        """
        while traci.simulation.getMinExpectedNumber() > 0: # type: ignore
            traci.simulationStep()
            simulation_time = traci.simulation.getTime()
            vehicles_in_simulation = traci.vehicle.getIDList()
            for vehicle_id in vehicles_in_simulation:
                # Get the edge id
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                if edge_id in MAJOR_EDGES:
                    veh = Vehicle(vehicle_id)
                    veh.add_road_type("major")
                elif edge_id in MINOR_EDGES:
                    veh = Vehicle(vehicle_id)
                    veh.add_road_type("minor")
                else:
                    continue

                
                x, y = traci.vehicle.getPosition(vehicle_id)
                lon, lot = traci.simulation.convertGeo(x, y)
                speed = traci.vehicle.getSpeed(vehicle_id)
                acc = traci.vehicle.getAcceleration(vehicle_id)
                acc_lon, acc_lat = veh.get_acc_lat_lon(acc)
                veh.add_lon(lon)
                veh.add_lat(lot)
                veh.add_time(simulation_time)
                veh.add_speed(speed)
                veh.add_acc_lon(acc_lon)
                veh.add_acc_lat(acc_lat)
            


                
            
            # Get the traffic lights
            tls = traci.trafficlight.getIDList()
            for tls_id in tls:
                if tls_id in TLS:
                    tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    tls_phase = traci.trafficlight.getPhase(tls_id)
                    print(f"Traffic light {tls_id} is in phase {tls_phase} with state {tls_state}")
                    



                

if __name__ == "__main__":
    generator = Generator()
    generator.start_sim()
    generator.run()

    # generator.generate_trips()
    # generator.generate_net()