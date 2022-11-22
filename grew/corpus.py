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
        if local:
            self._local = True
            if isinstance (data, dict):
                self.items = data
            else:
                dico = network.send_and_receive({"command": "corpus_get_all", "corpus_index": self.id })
                self.items = {sid: Graph(json_data) for (sid,json_data) in dico.items() }
        else:
            self._local = False
        # TODO: if data is a dict, requests corpus_get_all can be skipped

    def __len__(self):
        return len(self.sent_ids)

    def __setitem__(self,x,v):
        if self._local:
            self.items[x] = v
        else:
            self.update({x:v})

    def _get_one_item(self,sent_id):
        if self._local:
            return self.items[sent_id]
        else:
            req = {"command": "corpus_get", "corpus_index": self.id, "sent_id": sent_id}
            return (Graph(network.send_and_receive(req)))

    def __getitem__(self, data):
        """
        Search for [data] in previously loaded corpus
        :param data: a sent_id (type string) or a position (type int)
        :param corpus_index: an integer given by the [corpus] function
        :return: a graph
        """
        if isinstance(data, str):
            return self._get_one_item(data)
        if isinstance(data, int):
            return self._get_one_item(self.sent_ids[data])
        if isinstance(data, slice):
            return [self._get_one_item(self.sent_ids[i]) for i in range(data)]

    def _synchronize(self):
        self.update(self.items)

    def __iter__(self):
        return iter(self.sent_ids)

    def search(self,request,clustering_keys=[]):
        """
        Search for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [request] into the corpus
        """
        if self._local:
            ...
        else:
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
        if self._local:
            ...
        else:
            return network.send_and_receive({
                "command": "corpus_count",
                "corpus_index": self.id,
                "request": request.json_data(),
                "clustering_keys": clustering_keys,
            })

    def map(self, app, inplace=False):
        x = {sid : map(self[sid]) for sid in self}
        #load(x)

    def update(self, dico):
        req = {
            "command": "corpus_update",
            "corpus_index": self.id,
            "graphs": { sent_id : dico[sent_id].json_data() for sent_id in dico }
            }
        send_and_receive(req)

    def delete(self, key_list):
        ...


    def init(self, dictionnaire):
        pass

    def run(self, grs, strat : str="main"):
        pass
