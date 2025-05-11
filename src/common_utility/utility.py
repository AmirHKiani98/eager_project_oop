"""
Converts keys in a dictionary from strings to floats.
"""
def convert_keys_to_float(d):
    """
    Convert keys in a dictionary from strings to floats.
    Args:
        d (dict): The dictionary to convert.
    Returns:
        dict: The dictionary with keys converted to floats.
    """
    if isinstance(d, dict):
        return {float(k): convert_keys_to_float(v) for k, v in d.items() if k.replace('.', '', 1).isdigit()}
    return d
