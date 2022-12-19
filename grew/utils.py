import re
import math
import json

''' Graph utility tools '''

class GrewError(Exception):
    """A wrapper for grew-related errors"""

    def __init__(self, message):
        self.value = message
    def __str__(self):
        if isinstance(self.value, dict):
            return ("\n".join (("", "-"*80, json.dumps(self.value, indent=2), "-"*80)))
        else:
            return ("\n".join (("", "="*80, str (self.value), "="*80)))

GrewError.__doc__ = "A wrapper for grew-related errors"


def map_append(d,k,v):
    """
    append v to d[k] as a list
    """
    if k not in d:
        d[k] = []
    d[k].append(v)
