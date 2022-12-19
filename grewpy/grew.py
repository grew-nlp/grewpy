"""
Grew module : anything you want to talk about graph rewriting systems
Graphs are represented by dictionaries (nodes, edges, meta informations)
GRS can be build via files, explicit constructions or even strings. See doc of GREW
"""
import typing

from . import network

''' Library tools '''

JSON = typing.Any #dict[str,typing.Any] | list[typing.Any] | str | int

def set_config(data):
    """
    Change the configuration used in the next exchanges
    See https://grew.fr/doc/graph/#edges for details about config
    """
    return network.send_and_receive({"command": "set_config", "config": data})

def request_counter():
    return network.request_counter
