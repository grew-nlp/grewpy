"""
Grew module : anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
import os.path
import tempfile
import json
import typing

from .network import send_and_receive
from .graph import Graph
from .utils import GrewError
from . import network


class Corpus():
    def __init__(self,data):
        """Load a corpus from a file of a string
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        if isinstance(data, list):
            req = { "command": "load_corpus", "files": data }
            reply = network.send_request(req)
        elif os.path.isfile(data):
            req = { "command": "load_corpus", "files": [data] }
            reply = network.send_request(req)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".conll") as f:
                f.write(data)
                f.flush()  # to be read by others
                req = { "command": "load_corpus", "files": [f.name] }
                try:
                    reply = network.send_request(req)
                except GrewError:
                    raise GrewError(data)
        self.id =reply["index"]
        req = {"command": "corpus_sent_ids", "corpus_index": self.id}
        self.sent_ids = network.send_request(req)

    def __len__(self):
        return len(self.sent_ids)

    def __getitem__(self, data):
        """
        Search for [data] in previously loaded corpus
        :param data: a sent_id (type string) or a position (type int)
        :param corpus_index: an integer given by the [corpus] function
        :return: a graph
        """
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

    def __iter__(self):
        return iter(self.sent_ids)


    def search(self,request,clustering_keys=[]):
        """
        Search for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [request] into the corpus
        """
        return network.send_request({
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
        return network.send_request({
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
