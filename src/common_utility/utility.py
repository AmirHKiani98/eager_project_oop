"""
Converts keys in a dictionary from strings to floats.
"""
def convert_keys_to_float(d):
    """
    Recursively convert dict keys from str to float, but only for dicts — not inside list or primitive data.
    """
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            try:
                new_key = float(k)
            except (ValueError, TypeError):
                new_key = k  # Keep original if it can't be converted
            new_dict[new_key] = convert_keys_to_float(v)
        return new_dict
    else:
        return d