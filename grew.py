"""
Grew module : anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
import os.path
import re
import copy
import tempfile
import json

#from grew import network
#from grew import utils
import network
import utils

''' Library tools '''

def init(dev=False):
    """
    Initialize connection to GREW library
    :return: the ouput of the subprocess.Popen command.
    """
    return network.init(dev)

class Pattern():
    def __init__(self, r):
        """
        r is the string representation of the pattern
        """
        self.pat = r

class Corpus():
    def __init__(self,data):
        """Load a corpus from a file of a string
        :param data: a file, a list of files or a CoNLL string representation of a corpus
        :return: an integer index for latter reference to the corpus
        :raise an error if the files was not correctly loaded
        """
        try:
            if isinstance(data, list):
                req = { "command": "load_corpus", "files": data }
                reply = network.send_and_receive(req)
            elif os.path.isfile(data):
                req = { "command": "load_corpus", "files": [data] }
                reply = network.send_and_receive(req)
            else:
                with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".conll") as f:
                    f.write(data)
                    f.seek(0)  # to be read by others
                    req = { "command": "load_corpus", "files": [f.name] }
                    reply = network.send_and_receive(req)
            self.id =reply["index"]
            req = {"command": "corpus_sent_ids", "corpus_index": self.id}
            self.sent_ids = network.send_and_receive(req)
        except utils.GrewError as e: 
            raise utils.GrewError({"function": "grew.corpus", "data": data, "message":e.value})
    
    def __len__(self):
        return len(self.sent_ids)

    def __getitem__(self, data):
        """
        Search for [data] in previously loaded corpus
        :param data: a sent_id (type string) or a position (type int)
        :param corpus_index: an integer given by the [corpus] function
        :return: a graph
        """
        if isinstance(data, int):
            req = {
            "command": "corpus_get",
            "corpus_index": self.id,
            "position": data % len(self),
            }
        elif isinstance(data, str):
            req = {
            "command": "corpus_get",
            "corpus_index": self.id,
            "sent_id": data,
            }
        else:
            raise utils.GrewError({"function": "grew.corpus_get",
                              "message": "unexpected data, should be int or str"})
        try:
            return network.send_and_receive(req)
        except utils.GrewError as e:
            raise utils.GrewError(
            {"function": "grew.corpus_get", "message": e.value})

    def __iter__(self):
        return iter(self.sent_ids)

    def search(self,pattern):
        """
        Search for [pattern] into [corpus_index]
        :param patten: a string pattern
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [pattern] into the corpus
        """
        try:
            req = {
            "command": "corpus_search",
            "corpus_index": self.id,
            "pattern": pattern.pat,
            }
            return network.send_and_receive(req)
        except utils.GrewError as e:
            raise utils.GrewError(
            {"function": "grew.corpus_search", "message": e.value})

    def count(self,pattern):
        """
        Count for [pattern] into [corpus_index]
        :param patten: a string pattern
        :param corpus_index: an integer given by the [corpus] function
        :return: the number of matching of [pattern] into the corpus
        """
        try:
            req = {
            "command": "corpus_count",
            "corpus_index": self.id,
            "pattern": pattern.pat,
            }
            return network.send_and_receive(req)
        except utils.GrewError as e:
            raise utils.GrewError(
            {"function": "grew.corpus_count", "message": e.value})

def search(pattern, gr):
    """
    Search for [pattern] into [gr]
    :param patten: a string pattern
    :param gr: the graph
    :return: the list of matching of [pattern] into [gr]
    """
    try:
        req = {
            "command": "search",
            "graph": json.dumps(gr),
            "pattern": pattern
        }
        reply = network.send_and_receive(req)
        return reply
    except utils.GrewError as e: 
        raise utils.GrewError({"function": "grew.search", "message":e.value})

def graph_svg(graph):
    req = {
        "command": "dep_to_svg",
        "graph": json.dumps(graph),
    }
    return network.send_and_receive(req)

def json_grs(rs):
    req = {
        "command": "json_grs",
        "grs_index": rs,
    }
    return network.send_and_receive(req)


def grs(data):
    """Load a grs stored in a file
    :param data: either a file name or a Grew string representation of a grs
    :return: an integer index for latter reference to the grs
    :raise an error if the file was not correctly loaded
    """
    try:
        if os.path.isfile(data):
            req = {"command": "load_grs", "filename": data}
            reply = network.send_and_receive(req)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".grs") as f:
                f.write(data)
                f.seek(0)  # to be read by others
                req = {"command": "load_grs", "filename": f.name}
                reply = network.send_and_receive(req)
        return reply["index"]
    except utils.GrewError as e:
        raise utils.GrewError(
            {"function": "grew.grs", "data": data, "message": e.value})


def run(grs_data, graph_data, strat="main"):
    """
    Apply rs or the last loaded one to [gr]
    :param grs_data: a graph rewriting system or a Grew string representation of a grs
    :param graph_data: the graph, either a str (in grew format) or a dict
    :param strat: the strategy (by default "main")
    :return: the list of rewritten graphs
    """
    try:
        if isinstance(grs_data, int):
            grs_index = grs_data
        else:
            grs_index = grs(grs_data)

        req = {
            "command": "run",
            "graph": json.dumps(graph_data),
            "grs_index": grs_index,
            "strat": strat
        }
        reply = network.send_and_receive(req)
        return utils.rm_dups(reply)
    except utils.GrewError as e:
        raise utils.GrewError(
            {"function": "grew.run", "strat": strat, "message": e.value})

def json_grs(grs_index):
    try:
        req = {
            "command": "json_grs",
            "grs_index": grs_index,
        }
        return network.send_and_receive(req)
    except utils.GrewError as e: 
        raise utils.GrewError({"function": "grew.json_grs", "message":e.value})

