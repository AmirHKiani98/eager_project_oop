from src.model.traffic_model import TrafficModel

class CTM(TrafficModel):
    """
    Class representing the Cell Transmission Model (CTM) for traffic flow simulation.
    """

    def __init__(self, num_cells, cell_length, max_flow, min_flow):
        """
        Initialize the CTM with the number of cells, cell length, maximum flow, and minimum flow.

        :param num_cells: Number of cells in the model.
        :param cell_length: Length of each cell.
        :param max_flow: Maximum flow rate for each cell.
        :param min_flow: Minimum flow rate for each cell.
        """
        super().__init__(num_cells, cell_length)
        self.max_flow = max_flow
        self.min_flow = min_flow
    
    def update_cell_status(time, segment_id, densities, outflows, ctm_params, entry_flow, traffic_lights_df, traffic_lights_dict_states): # Maz
        num_cells = len(densities)
        new_densities = densities.copy()
        new_outflows = outflows.copy()     # Maz
        cell_length = cell_lengths[segment_id]
        dt = ctm_params.get_time_step(cell_length)
        for i in range(num_cells):  # iterate over all cells
            if i == 0:      # first cell
                inflow = entry_flow # no inflow
            else:           # for all other cells: minimum of max flow and the flow from the previous cell
                inflow = min(ctm_params.max_flow(cell_length), ctm_params.free_flow_speed * densities[i-1] * dt, ctm_params.wave_speed * (ctm_params.get_jam_density(cell_length) - densities[i]) * dt)

            if i == num_cells - 1:  # last cell
                # check if there is a traffic light at the end of the segment
                if is_tl(segment_id, traffic_lights_df):
                    # check the status of the traffic light
                    if tl_status(time, segment_id, traffic_lights_df, traffic_lights_dict_states) == 1: # green light
                        outflow = min(ctm_params.max_flow(cell_length), ctm_params.free_flow_speed * densities[i] * dt, math.inf)
                        new_outflows[i] = outflow
                    else:
                        outflow = 0
                        new_outflows[i] = outflow
                else:
                    outflow = min(ctm_params.max_flow(cell_length), ctm_params.free_flow_speed * densities[i] * dt, math.inf)
                    new_outflows[i] = outflow # Maz
            else:               # for all other cells: minimum of max flow and the flow to the next cell
                outflow = min(ctm_params.max_flow(cell_length), ctm_params.free_flow_speed * densities[i] * dt, ctm_params.wave_speed * (ctm_params.get_jam_density(cell_length) - densities[i+1]) * dt)
                new_outflows[i] = outflow # Maz
            new_densities[i] = densities[i] + (inflow - outflow) / cell_length   # n(t+1) = n(t) + (y(i) - y(i+1))/dx

        return new_densities, new_outflows # Maz