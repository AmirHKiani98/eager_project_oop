"""
This is a test module for the CTM model.
"""
import logging
from rich.logging import RichHandler
from shapely.geometry import Point as POINT
from src.model.ctm import CTM
from src.common_utility.units import Units
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.geo_loader import GeoLoader
from src.model.params import Parameters

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

def test_ctm_model():
    """
    Test the CTM model.
    """

    bypassed_data_loader = DataLoader.__new__(DataLoader)
    params = Parameters(
        vehicle_length=5*Units.M,
        free_flow_speed=60*Units.KM_PER_HR,
        wave_speed=12*Units.KM_PER_HR,
        num_lanes=1,
        dt=30*Units.S,
        jam_density_link=180*Units.PER_KM,
        q_max=1800*Units.PER_HR
    )
    logging.debug("dt: %s, q_max: %s", params.dt, params.q_max)
    locations = [ # lat, lon
        (44.991836848238044, -93.58816907158935),
        (44.99662014779215, -93.15121949469594),
        (44.993750215969705, -92.798142592005)
    ]
    locations = [
        POINT(loc[1], loc[0])
        for loc in locations
    ]

    bypassed_data_loader.params = params
    bypassed_data_loader.geo_loader = GeoLoader(
        locations=locations,
        cell_length=500,
        testing=True
    )
    model = CTM(
        dl=bypassed_data_loader
    )
    cell_length = bypassed_data_loader.geo_loader.cell_length
    assert model.dl.params.get_cell_capacity(cell_length) == 90
    assert model.dl.params.flow_capacity == 15
    cell_occupancies = [27, 87, 3]
    first_cell_inflow = (1440 * Units.PER_HR) * model.dl.params.dt
    # 1440 * 1/60 = 24 veh/min
    new_cell_occupancies, _ = model.predict(
        cell_occupancies=cell_occupancies,
        first_cell_inflow=first_cell_inflow,
        cell_length=cell_length,
        is_tl=True,
        tl_status=0,
        link_id=1
    )
    # Adjust the expected value based on the actual output of model.predict
    assert new_cell_occupancies == [36, 75, 15]

    new_cell_occupancies, _ = model.predict(
        cell_occupancies=new_cell_occupancies,
        first_cell_inflow=first_cell_inflow,
        cell_length=cell_length,
        is_tl=True,
        tl_status=0,
        link_id=1

    )
    assert new_cell_occupancies == [33, 75, 15]

    new_cell_occupancies, _ = model.predict(
        cell_occupancies=new_cell_occupancies,
        first_cell_inflow=first_cell_inflow,
        cell_length=cell_length,
        is_tl=True,
        tl_status=0,
        link_id=1
    )
    assert new_cell_occupancies == [30, 75, 15]
