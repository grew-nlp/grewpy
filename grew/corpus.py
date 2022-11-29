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


class Corpus(dict):
    def __init__(self,data):
        """Load a corpus from a file of a string
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :param local: state whether we load a local copy of each graph of the corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        acorpus = data if isinstance(data, AbstractCorpus) else AbstractCorpus(data)
        self._sent_ids = acorpus.get_sent_ids()
        super().__init__(acorpus.get_all())

    def __len__(self):
        return len(self._sent_ids)

    def __getitem__(self, data):
        """
        Search for [data] in previously loaded corpus
        :param data: a sent_id (type string) or a position (type int)
        :param corpus_index: an integer given by the [corpus] function
        :return: a graph
        """
        if isinstance(data, str):
            return super().__getitem__(data)
        if isinstance(data, int):
            return self[self._sent_ids[data]]
        if isinstance(data, slice):
            return [self[sid] for sid in self._sent_ids[data]]

    def __iter__(self):
        return iter(self._sent_ids)


class AbstractCorpus:
    def __init__(self, data, local=True):
        """Load a corpus on the CAML server
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :param local: state whether we load a local copy of each graph of the corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        if isinstance(data, list):
            req = {"command": "load_corpus", "files": data}
            reply = network.send_and_receive(req)
        elif isinstance(data, dict):
            req = {"command": "corpus_from_dict", "graphs": {
                sent_id: graph.json_data() for (sent_id, graph) in data.items()}}
            reply = network.send_and_receive(req)
        elif os.path.isfile(data):
            req = {"command": "load_corpus", "files": [data]}
            reply = network.send_and_receive(req)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".conll") as f:
                f.write(data)
                f.flush()  # to be read by others
                req = {"command": "load_corpus", "files": [f.name]}
                try:
                    reply = network.send_and_receive(req)
                except GrewError:
                    raise GrewError(data)
        self._id = reply["index"]

    def get_sent_ids(self):
        req = {"command": "corpus_sent_ids", "corpus_index": self._id}
        return network.send_and_receive(req)


    def get(self, sent_id):
        req = {"command": "corpus_get",
                   "corpus_index": self._id, "sent_id": sent_id}
        return (Graph(network.send_and_receive(req)))
    
    def __getitem__(self, data):
        if isinstance(data, str):
            return self.get(data)
        if isinstance(data, int):
            sids = self.get_sent_ids()
            return self.get(sids[data])
        if isinstance(data, slice):
            sids = self.get_sent_ids()
            return [self[sid] for sid in sids]


    def get_all(self):
        dico = network.send_and_receive({"command": "corpus_get_all", "corpus_index": self._id})
        return {sid: Graph(json_data) for (sid,json_data) in dico.items() }


    def search(self, request, clustering_keys=[]):
        """
        Search for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [request] into the corpus
        """
        return network.send_and_receive({
            "command": "corpus_search",
            "corpus_index": self._id,
            "request": request.json_data(),
            "clustering_keys": clustering_keys,
        })

    def count(self, request, clustering_keys=[]):
        """
        Count for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the number of matching of [request] into the corpus
        """
        return network.send_and_receive({
            "command": "corpus_count",
            "corpus_index": self._id,
            "request": request.json_data(),
            "clustering_keys": clustering_keys,
        })

    def __len__(self):
        #TODO
        return len(self.get_sent_ids())
