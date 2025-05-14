
class TrafficLight:
    all_tls = {}
    def __init__(self, tls_id, link_id):
        """
        Initialize a traffic light object.
        """
        self.tls_id = tls_id
        self.states = []
        self.times = []
        self.link_id = link_id

    def add_state(self, state):
        """
        Add a state to the traffic light.
        """
        self.states.append(state)
    
    def add_time(self, time):
        """
        Add a time to the traffic light.
        """
        self.times.append(time)

    def get_tls_processed_data(self):
        """
        Generate the processed data for the traffic light.
        Should follow the following format:
        trajectory_time,traffic_light_status,loc_link_id
        """
        text = ""
        for i in range(len(self.states)):
            text += f"{self.times[i]},{self.states[i]},{self.link_id}\n"
        return text
    
    @staticmethod
    def get_or_create_tls(tls_id, link_id) -> 'TrafficLight':
        """
        Get or create a traffic light object.
        """
        if tls_id not in TrafficLight.all_tls:
            TrafficLight.all_tls[tls_id] = TrafficLight(tls_id, link_id)
        return TrafficLight.all_tls[tls_id]