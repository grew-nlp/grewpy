"""
Grew module: anything you want to talk about graphs
"""
import os.path
import re
import copy
import tempfile
import json
import numpy as np
import warnings

from grewpy.grew import GrewError
from grewpy.network import send_and_receive
from grewpy import utils
from . import network

from collections import OrderedDict

# config is used to enocde the mapping from short/compact string representation of
# edge feature structures and the long representation as a dictionary like string
# See https://grew.fr/doc/graph/#edges

# A config if a couple of:
# - a string which is the basic feature name (the part without a prefix symbol)
# - an ordered dict associating other features names to their serapator.
# for now, mSUD config is hard-coded. TODO: make the config parametrized
sud_config = ('1', OrderedDict([('2', ':'), ('deep', '@'), ('type', '/')]))

''' interfaces'''
class FsEdge(dict):
    def __init__(self,data):
        if isinstance(data,str):
            super().__init__(FsEdge.parse(data))
        elif isinstance(data, dict):
            super().__init__(data)
        else:
            raise ValueError(f"data is not a feature structure {data}")

    def __eq__(self, other):
        for k,v in self.items():
            if k not in other or v != other[k]:
                return False
        return len(self) == len(other)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple((sorted(self.items()))))

    def __str__(self):
        return super().__repr__()

    def __repr__(self):
        return f"FsEdge({str(self)})"


    def compact(self):
        """
        sud style
        """
        self_keys = set(self.keys())
        if self_keys.issubset(set(sud_config[1].keys()) | {sud_config[0]}):
            s = "".join(f"{sep}{self[k]}" for k, sep in sud_config[1].items() if k in self)
            return f"{self.get(sud_config[0],'')}{s}"
        else:
            return str(self)

    @staticmethod
    def parse(s):
        """
        convert a string into a dictionary to be used by the constructor.
        2 cases:
         - s is a dict like string f=v,g=w…
         - s in a compact string, like "comp:aux@tense/m"
        return `GrewError` on ill formed input like "f=v,x"
        """
        if '=' in s: # s is parse as a dict like string f=v,g=w…
            clauses = dict()
            for item in s.split(","):
                if '=' in item:
                    a,b = item.split("=", maxsplit=1)
                    clauses[a] = b
                else:
                    raise GrewError(f"Cannot build FsEdge with data: {s}")
            return clauses
        else: # s is parsed following config
            clauses = dict()
            for key,separator in reversed(sud_config[1].items()):
                if separator in s:
                    s,v = s.rsplit(separator, 1)
                    clauses[key] = v
            clauses[sud_config[0]] = s
            return clauses

class Fs_edge(FsEdge):
    def __init__(self, X, f):
        warnings.warn(
            """Fs_edge is deprecated and will be removed in a future version.
            Please use FsEdge instead.
            See https://grew.fr/grewpy/upgrade_0.6/
            """,
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(X, f)



class Graph():
    """
    a dict mapping node keys to feature structure

    with extra data:
        - an extra dict `sucs` mapping node keys to successors (pair of edge feature,node key)
        - the list `order` containing nodes linearly ordered
        - and the `meta`(data) as a dict

    Param data: either
        - None: return an empty graph
        - a json formatted string
        - a file name containing a json/conll
        - a Graph: return a copy of the graph
        - or named arguments: `features`, `sucs`, `meta` and `order`
    """
    def __init__(self,data=None, **kwargs):

        if isinstance(data, Graph):
            self.features = dict(data.features)
            self._sucs = dict(data._sucs)
            self.meta = dict(data.meta)
            self.order = list(data.order)
        elif data is None:
            self.features = kwargs.get("features", dict())
            self.order = kwargs.get("order", [])
            self.meta = kwargs.get("meta", dict())
            self._sucs = kwargs.get("sucs", dict())
        elif isinstance(data, dict):
            (features, sucs, meta, order) = Graph._from_json(data)
            self.features = features
            self.order = order
            self.meta = meta
            self._sucs = sucs
        elif isinstance(data,str):
            #either filename, json or conll
            if os.path.isfile(data):
                req = {"command": "graph_load", "file": data}
                data_json = network.send_and_receive(req)
            else:
                try:
                    data_json = json.loads(data)
                except json.decoder.JSONDecodeError:
                    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".conll") as f:
                        f.write(data)
                        f.flush()  # to be read by others
                        req = {"command": "graph_load", "file": f.name}
                        data_json = network.send_and_receive(req)
            (self.features, self.sucs, self.meta, self.order) = Graph._from_json(data_json)
        else:
            raise GrewError(f"Cannot build Graph with data of type {type(data)}")
        assert isinstance(self.features,dict)

    @staticmethod
    def _from_json(data_json):
        features = data_json["nodes"]
        sucs = dict()
        for edge in data_json.get("edges", []):
            # TODO gestion des "label" implicite
            utils.map_append(sucs, edge["src"],
                             (edge["tar"], FsEdge(edge["label"])))
        meta = data_json.get("meta", dict())
        order = data_json.get("order", list())
        return (features, sucs, meta, order)

    @classmethod
    def from_json(cls,data_json):
        (features, sucs, meta, order) = Graph._from_json(data_json)
        return cls(features=features, sucs=sucs, order=order, meta=meta)

    def __len__(self):
        """
        return the number of nodes in self
        """
        return len(self.features)

    def __getitem__(self, nid):
        """
        return feature structure corresponding to nid
        """
        return (self.features[nid])

    def __iter__(self):
        return iter(self.features)

    def _gsucs(self):
        return self._sucs

    def _ssucs(self, v):
        self._sucs = v

    def _dsucs(self):
        self._sucs.clear()

    sucs = property(_gsucs, _ssucs, _dsucs, "successor relation")

    def json_data(self):
        nds = {c:self[c] for c in self.features}
        edg_list = []
        for n in self._sucs:
            for (e,s) in self._sucs[n]:
                if len(s.keys()) == 1 and '1' in s.keys():
                    s = s["1"]
                edg_list.append({"src":f"{n}", "label":s,"tar":f"{e}"})
        return {"nodes" : nds, "edges" : edg_list, "order": self.order, "meta" : self.meta }

    def __str__(self):
        return json.dumps(self.json_data(), indent=2)

    def to_conll(self):
        """
        return a CoNLL string for the given graph
        """
        data = self.json_data()
        req = {"command": "graph_to_conll", "graph": data}
        reply = send_and_receive(req)
        return reply

    def to_dot(self):
        """
        return a CoNLL string for the given graph
        """
        data = self.json_data()
        req = {"command": "graph_to_dot", "graph": data}
        reply = send_and_receive(req)
        return reply

    def to_svg(self, deco=None, draw_root=False):
        """
        return a SVG code for the given graph
        """
        data = self.json_data()
        req = {"command": "graph_to_svg", "graph": data, "deco": deco, "draw_root": draw_root}
        reply = send_and_receive(req)
        return reply

    def to_sentence(self, deco=None):
        """
        return a SVG code for the given graph
        """
        data = self.json_data()
        req = {"command": "graph_to_sentence", "graph": data, "deco": deco}
        reply = send_and_receive(req)
        return reply

    def triples(self):
        """
        return the list of edges presented as triples (n,e,s) with n-[e]-> s
        """
        return list((n, e, s) for n in self._sucs for s,e in self._sucs[n])

    def from_triples(self, triples):
        for n in self:
            self._sucs[n] = []
        for (n,e,m) in triples:
            self._sucs[n].append((e,m))

    def edge(self, n, m):
        """
        given node n and m
        return the "first" label of an edge between n and m if it exists
        """
        if n in self._sucs:
            for (k,v) in self._sucs[n]:
                if k == m:
                    return v
        return None

    def edge_up_to(self, n, m, criterion):
        if n in self._sucs:
            for k,v in self._sucs[n]:
                if k == m and criterion(v):
                    return v

    def edges(self, n, m):
        """
        given node n and m,
        return the set of edges between n and m
        """
        if n not in self._sucs:
            return []
        return [v for (k,v) in self._sucs[n] if k == m]

    def edges_up_to(self, n, m, criterion):
        """
        search for edges between n and m verifying some criterion
        """
        return [v for (k, v) in self._sucs[n] if k == m and criterion(v)]


    def run(self, Grs, strat="main"):
        return Grs.run(self, strat)

    def apply(self, Grs, strat="main"):
        return Grs.apply(self, strat)

    def edge_diff(self, other, edge_criterion=lambda e: True) -> np.array:
        """
        edge difference between two graphs
        """
        E1 = {(m, repr(e), n) for (m,e,n) in self.triples()  if edge_criterion(e)}  # set of edges as triples
        E2 = {(m, repr(e), n) for (m,e,n) in other.triples() if edge_criterion(e)}  # set of edges as triples
        return np.array([len(E1 & E2), len(E1 - E2), len(E2 - E1)])

    def lower(self, n, m):
        """
        given node n and m in g:
        return True if n < m in g
        """
        if n in self.order and m in self.order:
            return self.order.index(n) < self.order.index(m)
        return False

    def greater(self, n, s):
        """
        return True if n > s in g
        """
        if n in self.order and s in self.order:
            return self.order.index(n) > self.order.index(s)
        return False

    def edge_diff_up_to(self, other, edge_transform=lambda e:e):
        E1 = set()
        for m, e, n in self.triples():
            et = edge_transform(e)
            if et:
                E1.add((m,et,n))
        E2 = set()
        for m, e, n in other.triples():
            et = edge_transform(e)
            if et:
                E2.add((m, et, n))
        return np.array([len(E1 & E2), len(E1 - E2), len(E2 - E1)])

