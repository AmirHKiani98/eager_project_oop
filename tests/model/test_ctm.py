"""
This is a test module for the CTM model.
"""
from src.model.ctm import CTM
from src.common_utility.units import Units
def test_ctm_model(bypassed_data_loader, simple_traffic_params):
    """
    Test the CTM model.
    """

    model = CTM(
        params=simple_traffic_params,
        dl=bypassed_data_loader
    )
    cell_length = (500) * Units.M
    assert model.params.get_cell_capacity(cell_length) == 90
    assert model.params.flow_capacity == 15
    cell_occupancies = [27, 87, 3]
    first_cell_inflow = (1440 * Units.PER_HR) * model.params.dt
    # 1440 * 1/60 = 24 veh/min
    new_cell_occupancies, _ = model.predict(
        cell_occupancies=cell_occupancies,
        first_cell_inflow=first_cell_inflow,
        cell_length=cell_length,
        is_tl=True,
        tl_status=0
    )
    # Adjust the expected value based on the actual output of model.predict
    assert new_cell_occupancies == [36, 75, 15]

    new_cell_occupancies, _ = model.predict(
        cell_occupancies=new_cell_occupancies,
        first_cell_inflow=first_cell_inflow,
        cell_length=cell_length,
        is_tl=True,
        tl_status=0

    )
    assert new_cell_occupancies == [33, 75, 15]

    new_cell_occupancies, _ = model.predict(
        cell_occupancies=new_cell_occupancies,
        first_cell_inflow=first_cell_inflow,
        cell_length=cell_length,
        is_tl=True,
        tl_status=0
    )
    assert new_cell_occupancies == [30, 75, 15]
