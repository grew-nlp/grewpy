"""
Grew module : anything you want to talk about graph rewriting systems
Graphs are represented by dictionaries (nodes, edges, meta informations)
GRS can be build via files, explicit constructions or even strings. See doc of GREW
"""
import typing

from .graph import Graph
from .utils import GrewError

from . import network

''' Library tools '''

JSON = dict[str,typing.Any] | list[typing.Any] | str | int

def set_config(data):
    """
    Change the configuration used in the next exchanges
    See https://grew.fr/doc/graph/#edges for details about config
    """
    try:
        req = { "command": "set_config", "config": data }
        reply = network.send_and_receive(req)
        return reply
    except GrewError as e:
        raise GrewError({"function": "grew.set_config", "data":data, "message":e.value})

