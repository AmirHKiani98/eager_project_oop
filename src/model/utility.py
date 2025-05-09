"""
This module defines custom units for vehicle counts per kilometer and per hour.

"""
def run_wrapper(model, kwargs):
    """
    A wrapper function to run a traffic model with the provided arguments.
    Args:
        model (object): The traffic model instance to run.
        kwargs (dict): A dictionary of keyword arguments to pass to the model's run method.
    Returns:
        object: The result of the model's run method.
    """
    return model.run(**kwargs)
