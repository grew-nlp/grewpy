"""
Grew module : anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
import os.path
import glob

import sys
import tempfile
import json
import typing
import numpy as np

from .network import send_and_receive
from .graph import Graph
from .grew import GrewError
from .observation import Observation
from . import network

from .matchings import Matchings

class AbstractCorpus():
    def diff(self, other, edge_criterion=lambda e: True):
        """
        given two corpora, outputs the number of common edges, only left ones and only right ones
        """
        (common, left, right) = np.sum(
            [self[sid].diff(other[sid],edge_criterion) for sid in self], axis=0)
        precision = common / (common + left+1e-10)
        recall = common / (common + right+1e-10)
        f_measure = 2*precision*recall / (precision+recall+1e-10)
        return {
            "common": common,
            "left": left,
            "right": right,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f_measure": round(f_measure, 3),
        }

    def edge_diff_up_to(self, other, edge_transform=lambda e: e):
        (common, left, right) = np.sum(
            [self[sid].edge_diff_up_to(other[sid], edge_transform) for sid in self], axis=0)
        precision = common / (common + left)
        recall = common / (common + right)
        f_measure = 2*precision*recall / (precision+recall)
        return {
            "common": common,
            "left": left,
            "right": right,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f_measure": round(f_measure, 3),
        }

class CorpusDraft(AbstractCorpus,dict):
    """
    the draft is composed of 
      - self, a dict mapping sentence_id to graphs
      - self._sent_ids, a list that specifies the sentence order
    """
    def __init__(self,data=None):
        """Load a corpus from a file of a string
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        if isinstance(data, CorpusDraft):
            T = {sid: Graph(data[sid]) for sid in data}
            super().__init__(T)
        elif isinstance(data, dict):
            super().__init__(data)
        elif data == None:
            super().__init__()
        else:
            acorpus = data if isinstance(data, Corpus) else Corpus(data)
            self._sent_ids = acorpus.get_sent_ids() #specifies the sentences order
            super().__init__(acorpus.get_all())

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

    def apply(self, fun):
        """
        Apply fun to all graphs, return the new Corpus
        """
        return CorpusDraft({sid : fun(self[sid]) for sid in self})


class Corpus(AbstractCorpus):
    def __init__(self, data):
        """An abstract corpus
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        if isinstance(data, list):
            if data and isinstance(data[0], Graph):
                graphs = {f'{i}' : data[i].json_data() for i in range(len(data))}
                req = {"command": "corpus_from_dict", "graphs": graphs}
            else:
                #supposed to be a list of files
                req = {"command": "corpus_load", "files": data}
            reply = network.send_and_receive(req)
        elif isinstance(data, dict):
            req = {"command": "corpus_from_dict", "graphs": {
                sent_id: graph.json_data() for (sent_id, graph) in data.items()}}
            reply = network.send_and_receive(req)
        elif os.path.isdir(data):
            # load of connlu files of the directory
            file_list = glob.glob(f"{data}/*.conllu")
            req = {"command": "corpus_load", "files": file_list}
            reply = network.send_and_receive(req)
        elif os.path.isfile(data):
            req = {"command": "corpus_load", "files": [data]}
            reply = network.send_and_receive(req)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".conll") as f:
                f.write(data)
                f.flush()  # to be read by others
                req = {"command": "corpus_load", "files": [f.name]}
                try:
                    reply = network.send_and_receive(req)
                except GrewError:
                    raise GrewError(data)
        self._length = reply["length"]
        self._id = reply["index"]

    def get_sent_ids(self):
        """
        return the list of sentence ids
        """
        req = {"command": "corpus_sent_ids", "corpus_index": self._id}
        return network.send_and_receive(req)

    def get_id(self):
        """
        return the id of the corpus
        """
        return self._id

    def clean(self):
        """
        clean the corpus (remove from the backend memory)
        """
        req = {"command": "corpus_clean", "corpus_index": self._id}
        return network.send_and_receive(req)

    def get(self, sent_id):
        """
        return a graph corresponding to the sentence id sent_id
        """
        req = {"command": "corpus_get",
                   "corpus_index": self._id, "sent_id": sent_id}
        return (Graph.from_json(network.send_and_receive(req)))
    
    def __getitem__(self, data):
        """
        return a graph corresponding to data, either
          - a sentence id,
          - an index in the sentence id array
          - a slice
        """
        if isinstance(data, str):
            return self.get(data)
        if isinstance(data, int):
            sids = self.get_sent_ids()
            return self.get(sids[data])
        if isinstance(data, slice):
            sids = self.get_sent_ids()
            return [self[sid] for sid in sids[data]]


    def get_all(self):
        """
        return a dictionary mapping sentence ids to graphs
        """
        dico = network.send_and_receive({"command": "corpus_get_all", "corpus_index": self._id})
        return {sid: Graph.from_json(json_data) for (sid,json_data) in dico.items() }


    def search(self, request, clustering_parameter=[], clustering_keys=[],flat=None):
        """
        Search for [request] into [corpus_index]

        Parameters:
        request (Request): a request
        corpus_index: an integer given by the [corpus] function

        Returns:
        list: the list of matching of [request] into the corpus
        """
        res = network.send_and_receive({
            "command": "corpus_search",
            "corpus_index": self._id,
            "request": request.json_data(),
            "clustering_keys": clustering_parameter + clustering_keys
        })
        if flat == "matchings":
            return Matchings(res, self)
        elif flat == "observations" and clustering_parameter or clustering_keys:
            return Observation(res, clustering_parameter, clustering_keys)
        return res

    def count(self, request, clustering_parameter=[], clustering_keys=[], flat=False):
        """
        Count for [request] into [corpus_index]
        :param request: a string request
        :param corpus_index: an integer given by the [corpus] function
        :return: the number of matching of [request] into the corpus
        """
        res = network.send_and_receive({
            "command": "corpus_count",
            "corpus_index": self._id,
            "request": request.json_data(),
            "clustering_keys": clustering_parameter + clustering_keys,
        })
        if not flat:
            return res
        if clustering_parameter or clustering_keys:
            return Observation(obs=res,parameter=clustering_parameter, keys=clustering_keys)
        return res

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter(self.get_sent_ids())

    def to_conll(self):
        """
        return a CoNLL string for the current corpus
        """
        reply = network.send_and_receive({
            "command": "corpus_to_conll",
            "corpus_index": self._id
        })
        return reply
