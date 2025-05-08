"""

Point Queue module for managing traffic flow at intersections.
This module implements a queue system for vehicles waiting at traffic signals.
"""
import math
from src.model.traffic_model import TrafficModel

class PointQueue(TrafficModel):
    """
    Class representing a point queue for managing traffic flow at intersections.
    This class is designed to handle the queue of vehicles waiting at a traffic signal.
    """
    def __init__(self, geo_loader, params, dl, plotter):
        """
        Initialize the PointQueue with a GeoLoader instance, Parameters object,
        and a DataLoader instance.
        Args:
            geo_loader (GeoLoader): An instance of GeoLoader for geographical data.
            params (Parameters): An instance of Parameters for model configuration.
            dl (DataLoader): An instance of DataLoader for loading data.
            plotter (Plotter): An instance of Plotter for visualizing data.
        """
        super().__init__(geo_loader, params, dl, plotter)


    def predict(
        self,
        **kwargs,
    ):
        """
        Predict the traffic flow at a specific time step using the point queue model.
        Args:
            time (int): The current time step.
            link_id (int): The ID of the link.
            entry_flow (float): The flow of vehicles entering the cell.
            traffic_lights_df (DataFrame): DataFrame containing traffic light information.
            traffic_lights_dict_states (dict): Dictionary containing traffic light states.
        Returns:
            tuple: A tuple containing the updated link density and outflow.
        """
        required_keys = {
            "time",
            "link_id",
            "entry_flow",
            "traffic_lights_df",
            "traffic_lights_dict_states",
        }
        if not required_keys.issubset(kwargs):
            missing = required_keys - kwargs.keys()
            raise ValueError(f"Missing required parameters for PointQueue.predict(): {missing}")
    
        time = kwargs["time"]
        link_id = kwargs["link_id"]
        entry_flow = kwargs["entry_flow"]
        traffic_lights_df = kwargs["traffic_lights_df"]
        traffic_lights_dict_states = kwargs["traffic_lights_dict_states"]
        # check if the link_id is in the cell_lengths dictionarytime,
        # link_id,
        # entry_flow,
        # traffic_lights_df,
        # traffic_lights_dict_states

        cell_length = cell_lengths[link_id]

        segment_length = segments_gdf_exploded[
            segments_gdf_exploded["link_id"] == link_id
        ].iloc[0].length

        if is_tl(link_id, traffic_lights_df):
            # check the status of the traffic light
            if (
                tl_status(
                    time, link_id, traffic_lights_df, traffic_lights_dict_states
                ) == 1  # green light
            ):
                # find the link sending flow using point queue model
                sending_flow = min(
                    N_upstr(
                        time + ctm_params.get_time_step(cell_length)
                        - (segment_length / ctm_params.free_flow_speed),
                        link_id
                    ) - N_downstr(time, link_id),
                    ctm_params.max_flow(cell_length)
                    * ctm_params.get_time_step(cell_length)
                )
            else:
                sending_flow = 0
        else: # no traffic light at the end of the link
            sending_flow = min(
                N_upstr(
                    time + ctm_params.get_time_step(cell_length)
                    - (segment_length / ctm_params.free_flow_speed),
                    link_id
                ) - N_downstr(time, link_id),
                ctm_params.max_flow(cell_length) * ctm_params.get_time_step(cell_length)
            )

        # find the number of vehicles in the link at the next time step
        # current number of vehicles
        n_current = (
            N_upstr(time, link_id) - N_downstr(time, link_id)
        )

        n_updated = (
            n_current
            + entry_flow * ctm_params.get_time_step(cell_length)
            - sending_flow
        )
        if n_updated < 0:
            print(
                "n_current:", n_current,
                "sending_flow:", sending_flow,
                "entry_flow:", entry_flow,
                "time:", time
            )
            print(
                "N DOWNSTR is", N_upstr(time, link_id),
                "at this time N special is",
                N_upstr(
                    time + ctm_params.get_time_step(cell_length)
                    - (segment_length / ctm_params.free_flow_speed),
                    link_id
                )
            )

        link_outflow = sending_flow
        link_density = n_updated / segment_length
        return link_density, link_outflow
