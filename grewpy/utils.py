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
