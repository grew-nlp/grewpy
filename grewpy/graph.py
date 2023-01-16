"""
Grew module: anything you want to talk about graphs
"""
import os.path
import re
import copy
import tempfile
import json
import numpy as np

from grewpy.grew import GrewError
from grewpy.network import send_and_receive
from grewpy import utils
from . import network

''' interfaces'''
class Fs_edge(dict):
    def __init__(self,data):
        if isinstance(data,str):
            super().__init__({"1": data})
        elif isinstance(data, dict):
            super().__init__(data)
        else:
            raise ValueError(f"data is not a feature structure {data}")

    def __hash__(self):
        return (hash (str(self)))

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
            self.features = data.get("features", dict())
            self.order = data.get("order", [])
            self.meta = data.get("meta", dict())
            self._sucs = data.get("sucs", dict())
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
                             (edge["tar"], Fs_edge(edge["label"])))
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

    def to_dot(self): # TODO fix it
        """
        return a string in dot/graphviz format
        """
        s = 'digraph G{\n'
        for n,fs in self.features.items():
            s += f'{n}[label="'
            label = ["%s:%s" % (f,v.replace('"','\\"')) for f, v in fs.items()]
            s += ",".join(label)
            s += '"];\n'
        s += "\n".join([f'{n} -> {m}[label="{e}"];' for n,suc in self._sucs.items() for e,m in suc])
        return s + '\n}'

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
        return f"({str(self.features)}, {str(self._sucs)})" # TODO order, meta

    def to_conll(self):
        """
        return a CoNLL string for the given graph
        """
        data = self.json_data()
        req = {"command": "graph_to_conll", "graph": data}
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

    def edges(self, n, m):
        """
        given node n and m, 
        return the set of edges between n and m
        """
        return [v for (k,v) in self._sucs[n] if k == m]

    

    def run(self, Grs, strat="main"):
        return Grs.run(self, strat)

    def apply(self, Grs, strat="main"):
        return Grs.apply(self, strat)

    def diff(self, other, skip_edge_criterion=lambda e: False) -> np.array:
        E1 = {(m,repr(e),n) for (m,e,n) in self.triples() if not skip_edge_criterion(e)}  # set of edges as triples
        E2 = {(m, repr(e), n) for (m, e, n) in other.triples()
              if not skip_edge_criterion(e)}  # set of edges as triples
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
