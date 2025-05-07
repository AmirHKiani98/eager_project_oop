
class Parameters():

    def __init__(self,  vehicle_length=5, free_flow_speed=15, wave_speed=10, num_lanes=3):
        self.num_lanes = num_lanes
        self.vehicle_length = vehicle_length
        self.free_flow_speed = free_flow_speed
        self.wave_speed = wave_speed
        self.jam_density_link = 130 
        self.jam_density_FD = self.jam_density_link * self.num_lanes
        self.max_flow_link = 2000 / 3600 * self.num_lanes
        self.critical_density = 50 * self.num_lanes 
        self.tau = 1
        self.c = 10.14 
        self.c0 = self.free_flow_speed/(self.tau*2)
        
    def max_flow(self, cell_length):
        """
        Calculate the maximum flow in the system based on the fundamental diagram.

        Returns:
            float: Maximum flow (vehicles/second).
        """
        return min(1800, min(self.free_flow_speed, self.wave_speed) * self.get_jam_density(cell_length) * self.num_lanes)

    def get_time_step(self, cell_length):

        return cell_length / self.free_flow_speed

    def get_jam_density(self, cell_length):
         return self.jam_density_link/1000 * cell_length * self.num_lanes