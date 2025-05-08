"""
This is a test module for the CTM model.
"""
from src.model.ctm import CTM
def test_ctm_model(bypassed_data_loader, simple_traffic_params):
    """
    Test the CTM model.
    """

    model = CTM(
        params=simple_traffic_params,
        dl=bypassed_data_loader
    )
    model.predict(
        
    )
