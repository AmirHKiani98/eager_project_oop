"""
A module for gererating test data for unit tests.
This uses sumo library to generate a random network and then generates random trips
for the network. (Traci_
"""
import traci
from pathlib import Path
from tests.generating_data.vehicle import Vehicle
from tests.generating_data.traffic_light import TrafficLight
TLS = {
    "cluster_300400470_5415104194_6175368841_6175368842": "168914458#0",
    "GS_cluster_10134104320_250691755": "299510529#1",
    "GS_250691739": "299644700#2",
    "cluster_250691791_5697162223_6220792260_952119412": "299644700#8"
}

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
    """
    A class to generate test data for unit tests.
    This uses sumo library to generate a random network and then generates random trips
    for the network. (Traci_)
    The generated data is stored in the following format:
    track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time
    track_id; type; traveled_d; avg_speed are just one number
    lat; lon; speed; lon_acc; lat_acc groups of 6
    """
    def __init__(self, simulation_time = 800, cell_length=None, cell_numbers=None) -> None:
        self.base_dir = Path(__file__).parent.parent.parent
        self.net_file = self.base_dir / "tests" / "assets" / "sumo_net" / "osm.net.xml"
        self.route_file = self.base_dir / "tests" / "assets" / "sumo_net" / "trips.trips.xml"
        self.sumo_config_file = self.base_dir / "tests" / "assets" / "sumo_net" / "osm.sumocfg"
        self.simulation_time = simulation_time
        if cell_length is None and cell_numbers is None:
            raise ValueError(
                "You must provide either cell_length or cell_numbers to the generator."
            )

        self.cell_length = cell_length
        self.cell_numbers = cell_numbers

    def find_edges_lanes(self):
        """
        Find the edges and lanes in the network.
        """
        edges = traci.edge.getIDList()
        edges = {
            edge_id: [] for edge_id in edges
        }
        lanes = traci.lane.getIDList()
        for lane_id in lanes:
            edge_id = traci.lane.getEdgeID(lane_id)
            edges[edge_id].append(lane_id)
        return edges

    def generate_e3_detectors(self, edge_id, file_handle):
        """
        Generate E3 detectors for all lanes of the given edge into the shared file handle.
        """
        lane_ids = self.edges_lanes[edge_id]
        for lane_id in lane_ids:
            lane_length = traci.lane.getLength(lane_id)
            if self.cell_numbers is not None and isinstance(self.cell_numbers, int):
                cell_size = lane_length / self.cell_numbers # type: ignore
            else:
                cell_size = self.cell_length
            n_cells = int(lane_length / cell_size) # type: ignore
            print(n_cells)
            for i in range(n_cells):
                entry_pos = i * cell_size # type: ignore
                exit_pos = (i + 1) * cell_size # type: ignore
                detector_id = f"det_{edge_id}_{lane_id}_{i}"

                file_handle.write(f'  <entryExitDetector id="{detector_id}" period="1" file="e3_output.xml" timeThreshold="2.0" speedThreshold="5.0">\n')
                file_handle.write(f'    <detEntry lane="{lane_id}" pos="{entry_pos}" friendlyPos="true"/>\n')
                file_handle.write(f'    <detExit lane="{lane_id}" pos="{exit_pos}" friendlyPos="true"/>\n')
                file_handle.write(f'  </entryExitDetector>\n')
    
    def start_sim(self):
        """
        Start the simulation using the sumo library.
        """
        traci.start(["sumo", "-n", str(self.net_file), "-r", str(self.route_file), "--step-length", "0.04"])    
        self.edges_lanes = self.find_edges_lanes()
        

        # Create ONE single detector file
        detectors_file = self.base_dir / "tests" / "assets" / "sumo_net" / "e3_detectors.add.xml"
        with open(detectors_file, 'w') as f:
            f.write('<additional>\n')
            for edge_id in MAJOR_EDGES:
                if edge_id not in self.edges_lanes:
                    continue
                print("shit")
                self.generate_e3_detectors(edge_id, f)
            f.write('</additional>\n')
        traci.close()
        exit()
        
    def run(self):
        """
        Step through the simulation.
        """
        self.start_sim()
        simulation_time = 0
        while simulation_time < self.simulation_time: # type: ignore
            
            simulation_time = traci.simulation.getTime()

            vehicles_in_simulation = traci.vehicle.getIDList()
            for vehicle_id in vehicles_in_simulation:
                # Get the edge id
                
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                if edge_id in MAJOR_EDGES:
                    veh = Vehicle.get_or_create(vehicle_id)
                    veh.add_road_type("major")
                elif edge_id in MINOR_EDGES:
                    veh = Vehicle.get_or_create(vehicle_id)
                    veh.add_road_type("minor")
                else:
                    continue
                
                x, y = traci.vehicle.getPosition(vehicle_id)
                lon, lat = traci.simulation.convertGeo(x, y)
                veh.add_lon(lon)
                veh.add_lat(lat)
                speed = traci.vehicle.getSpeed(vehicle_id)
                acc = traci.vehicle.getAcceleration(vehicle_id)
                acc_lon, acc_lat = veh.get_acc_lat_lon(acc)

                veh.add_time(simulation_time)
                veh.add_speed(speed)
                veh.add_acc_lon(acc_lon)
                veh.add_acc_lat(acc_lat)
                veh.add_link_id(edge_id)

            # Get the traffic lights
            tls = traci.trafficlight.getIDList()
            for tls_id in tls:
                if tls_id in TLS.keys():
                    tls_obj = TrafficLight.get_or_create_tls(tls_id, TLS[tls_id])
                    tls_obj.add_time(simulation_time)
                    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                    tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    
                    if len(controlled_lanes) != len(tls_state):
                        raise ValueError(
                            f"Controlled lanes {controlled_lanes} and tls state {tls_state} do not match"
                        )
                    
                    # Go over each controlled lane
                    found_link = False
                    for idx, lane_id in enumerate(controlled_lanes):
                        # Extract the edge part (lane_id is like "299510529#0_0")
                        edge_id = lane_id.split("_")[0]

                        if edge_id == TLS[tls_id] and not found_link:
                            found_link = True
                            signal_color = str(tls_state)[idx]  # 'G', 'r', 'y', etc.
                            if signal_color == 'G':
                                tls_obj.add_state(1)
                            else:
                                tls_obj.add_state(0)
                    
            traci.simulationStep()
    
        traci.close()
        self.generate_files()
    
    def _generate_raw_data(self):
        """
        Generate the raw data that follows the same format as pneuma
        track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time
        track_id; type; traveled_d; avg_speed are just one number
        lat; lon; speed; lon_acc; lat_acc groups of 6
        """
        raw_data_text = "track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time\n"
        for vehicle_id, vehicle in Vehicle.all_vehicles.items():
            raw_data_text += vehicle.get_veh_raw_data() + "\n"
        
        return raw_data_text
    
    def _generate_ground_truth_data(self):
        """
        generate the ground truth that follows the format after processing the raw data
        """
        modified_data_text = "lat,lon,speed,lon_acc,lat_acc,trajectory_time,track_id,veh_type,traveled_d,avg_speed,link_id,distance_from_link\n"
        for vehicle_id, vehicle in Vehicle.all_vehicles.items():
            modified_data_text += vehicle.generate_processed_data()
        return modified_data_text

    def _generate_tls_ground_truth_data(self):
        """
        Generate the data for the traffic lights.
        """
        text = "trajectory_time,traffic_light_status,loc_link_id\n"
        for tls_id, tls in TrafficLight.all_tls.items():
            text += tls.get_tls_processed_data()
        return text

    def generate_files(self):
        """
        Generate final files
        """
        raw_data = self._generate_raw_data()
        processed_ground_truth = self._generate_ground_truth_data()
        tls_ground_truth = self._generate_tls_ground_truth_data()
        with open(self.base_dir / "tests" / "assets" / "raw_data.csv", "w") as f:
            f.write(raw_data)
        with open(self.base_dir / "tests" / "assets" / "processed_ground_truth.csv", "w") as f:
            f.write(processed_ground_truth)
        with open(self.base_dir / "tests" / "assets" / "tls_ground_truth.csv", "w") as f:
            f.write(tls_ground_truth)
