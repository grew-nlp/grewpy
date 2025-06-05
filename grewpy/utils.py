import re
import math
import json

''' Graph utility tools '''

def map_append(d,k,v):
    """
    append v to d[k] as a list
    """
    if k not in d:
        d[k] = []
    d[k].append(v)

def flatten_dict_keys(d, parent_key=()):
    """
    Flattens the keys of a nested dictionary into tuples.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (tuple): The current parent key for recursion.

    Returns:
        dict: A new dictionary with flattened tuple keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.extend(flatten_dict_keys(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)