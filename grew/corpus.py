"""
Grew module : anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
import os.path
import sys
import tempfile
import json
import typing

from .network import send_and_receive
from .graph import Graph
from .utils import GrewError
from . import network


class Corpus():
    def __init__(self,data, local=True):
        """Load a corpus from a file of a string
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :param local: state whether we load a local copy of each graph of the corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        if isinstance(data, list):
            req = { "command": "load_corpus", "files": data }
            reply = network.send_and_receive(req)
        elif isinstance(data, dict):
            req = { "command": "corpus_from_dict", "graphs": data }
            reply = network.send_and_receive(req)
        elif os.path.isfile(data):
            req = { "command": "load_corpus", "files": [data] }
            reply = network.send_and_receive(req)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".conll") as f:
                f.write(data)
                f.flush()  # to be read by others
                req = { "command": "load_corpus", "files": [f.name] }
                try:
                    reply = network.send_and_receive(req)
                except GrewError:
                    raise GrewError(data)
        self.id =reply["index"]
        req = {"command": "corpus_sent_ids", "corpus_index": self.id}
        self.sent_ids = network.send_and_receive(req)
        self.unsynchronized_sent_ids = set()
        if local:
            dico = network.send_and_receive({"command": "corpus_get_all", "corpus_index": self.id })
            self.items = {sid: Graph(json_data) for (sid,json_data) in dico.items() }
        else:
            self.items = dict() 
    # TODO: if data is a dict, requests corpus_get_all can be skipped

    def __len__(self):
        return len(self.sent_ids)

    def __setitem__(self,x,v):
        self.unsynchronized_sent_ids.add (x)
        self.items[x]=v

    def _get_one_item(self,sent_id):
        if sent_id in self.items:
            return self.items[sent_id]
        else:
            req = {"command": "corpus_get", "corpus_index": self.id, "sent_id": sent_id}
            graph = Graph(network.send_and_receive(req))
            self.items[sent_id] = graph
            self.unsynchronized_sent_ids.discard(sent_id)
            return graph

    def __getitem__(self, data):
        """
        Search for [data] in previously loaded corpus
        :param data: a sent_id (type string) or a position (type int)
        :param corpus_index: an integer given by the [corpus] function
        :return: a graph
        """
<<<<<<< HEAD
        if self.local:
            if isinstance(data, slice):
                start, stop, step = data.start or 0, data.stop or sys.maxsize, data.step or 1
                names = [n for n in self.sent_ids[start:stop:step]]
                return [self.items[n] for n in names]
            if isinstance(data, str):
                return self.items[data]
            if isinstance(data, int):
                return self.items[self.sent_ids[data]]
        else:
            req = {"command": "corpus_get", "corpus_index": self.id}
            if isinstance(data, slice):
                start = data.start if data.start else 0
                stop = data.stop if data.stop else len(self)
                step = data.step if data.step else 1
                res = []
                for i in range(start,stop, step):
                    req["position"] = i
                    res.append(Graph(network.send_and_receive(req)))
                return res
            if isinstance(data, int):
                req["position"]  = data % len(self)
            elif isinstance(data, str):
                req["sent_id"] =  data
            return Graph(network.send_and_receive(req))
=======
        if isinstance(data, str):
            return self._get_one_item(data)
        if isinstance(data, int):
            return self._get_one_item(self.sent_ids[data])
        if isinstance(data, slice):
            return [self._get_one_item(self.sent_ids[i]) for i in data]

    def _synchronize(self):
        req = {
            "command": "corpus_update",
            "corpus_index": self.id,
            "graphs": { sent_id : self.items[sent_id].json_data() for sent_id in self.unsynchronized_sent_ids }
            }
        send_and_receive(req)
        self.unsynchronized_sent_ids = set()
>>>>>>> ec7a54c9bd7f83e9b4cf13cd1e62c3ed728066b3

    def __iter__(self):
        return iter(self.sent_ids)

    def search(self,request,clustering_keys=[]):
        """
        Search for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [request] into the corpus
        """
        self._synchronize()
        return network.send_and_receive({
            "command": "corpus_search",
            "corpus_index": self.id,
            "request": request.json_data(),
            "clustering_keys": clustering_keys,
            })

    def count(self,request,clustering_keys=[]):
        """
        Count for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the number of matching of [request] into the corpus
        """
        self._synchronize()
        return network.send_and_receive({
            "command": "corpus_count",
            "corpus_index": self.id,
            "request": request.json_data(),
            "clustering_keys": clustering_keys,
            })

    def map(self, app, inplace=False):
        x = {sid : map(self[sid]) for sid in self}
        #load(x)

    def update(self, dict):
        pass

    def delete(self, key_list):
        ...


    def init(self, dictionnaire):
        pass

    def run(self, grs, strat : str="main"):
        pass
