"""
Grew python library
All you need to use grew 
See grew online documentation for global informations
"""
from .corpus import Corpus, AbstractCorpus
from .grs import Request, GRS, Package, Rule, Command, AbstractGRS
from .graph import Graph
from .network import init

from .grew import set_config

init(False)