"""
Grew module : anything you want to talk about graph rewriting systems
Graphs are represented by dictionaries (nodes, edges, meta informations)
GRS can be build via files, explicit constructions or even strings. See doc of GREW
"""
import typing
import json

from . import network

''' Library tools '''

JSON = typing.Any #dict[str,typing.Any] | list[typing.Any] | str | int

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

def set_config(data):
    """
    Change the configuration used in the next exchanges
    See https://grew.fr/doc/graph/#edges for details about config
    """
    return network.send_and_receive({"command": "set_config", "config": data})

def request_counter():
    return network.request_counter
