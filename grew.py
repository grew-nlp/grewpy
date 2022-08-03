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
from unicodedata import is_normalized

#from grew import network
#from grew import utils
import network
import utils
from graph import Graph

''' Library tools '''

def init(dev=False):
    """
    Initialize connection to GREW library
    :return: the ouput of the subprocess.Popen command.
    """
    return network.init(dev)
init()

class ClauseList(list):
    """
    a list of clauses
    """
    def __init__(self,*L):
        if len(L)>=2:
            self.sort = L[0]        
            if isinstance(L,str):
                super().__init__([c.strip() for c in L[1].split(";") if c.strip()])
            else:
                super().__init__(*L[1:])
        elif len(L) == 1 and isinstance(L[0],ClauseList):
            self = L[0]

    def json(self):
        return f"{self.sort}{{{';'.join(self)}}}"

class Pattern(list):
    def __init__(self, *L):
        """
        L is a list of ClauseList or pairs (sort,clauses)
        """
        super().__init__([C if isinstance(C,ClauseList) else ClauseList(*C) for C in L])

    def json(self):
        return "".join([C.json() for C in self])

class Command():
    def __init__(self, *L):
        self.cmds = L
    def json(self):
        cm = '\n'.join(self.cmds)
        return f"commands{{ {cm}}}"

class Rule():
    def __init__(self, name, pattern, cmds):
        self.pattern = pattern
        self.commands = cmds
        self.name = name
    def json(self):
        p = self.pattern.json()
        c = self.commands.json()
        return f"rule {self.name} {{ {p}\n {c} }}"

class Strategy:
    def __init__(self, json):
        self.name = json["strat_name"]
        self.data = AST(json["strat_def"])

    def json(self):
        return f"{{{self.name} : {self.data.json()} }}"

class GRS():

    def load_grs(data):
        """load data (either a filename or a json encoded string) within grew"""
        if not os.path.isfile(data):
            f = tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".grs")
            f.write(data)
            req = {"command": "load_grs", "filename": f.name}
        else:
            req = {"command": "load_grs", "filename": data}
        try:
            reply = network.send_and_receive(req)
            index = reply["index"]
            req = {"command": "json_grs", "grs_index": index}
            json = network.send_and_receive(req)
            return index,json
        except utils.GrewError as e:
            raise utils.GrewError(
                {"function": "grew.GRS", "data": data, "message": e.value})
    def __ii(self,n='',p=dict(),s=dict(),r=dict()):
        self.filename =n
        self.packages = p
        self.rules = r
        self.strats = s
        self.index = -1

    def __init__(self, data=None, **kwargs):
        """Load a grs stored in a file
        :param data: either a file name or a Grew string representation of a grs
        :return: an integer index for latter reference to the grs
        :raise an error if the file was not correctly loaded
        """
        if data is None:
            GRS.__ii(self)
            return
        elif len(kwargs)>0:
            #explicit declaration
            GRS.__ii(data, kwargs.get('packages',dict()),kwargs.get('rules',dict()),kwargs.get('strats',dict()))
        elif isinstance(data, str):
            index,json = GRS.load_grs(data)
            GRS.__ii(self,json['filename'])
            self.index = index
            for d in json["decls"]:
                if 'strat_name' in d:
                    utils.map_append(
                        self.strats, d['strat_name'], d['strat_def'])
                elif 'package_name' in d:
                    utils.map_append(
                            self.packages, d['package_name'], d['decls'])
                elif 'rule_name' in d:
                    utils.map_append(self.rules, d['rule_name'], d['rule'])
                else:
                    raise utils.GrewError(f"{d} is not part of a grs")
        else:
            pass
            """
            TO BE IMPLEMENTED
            """

    def json(self):
        sts = ", ".join(
            [f"{{'strat_name' : {s}, 'strat_def':{v}}}" for s, v in self.strats.items()])
        pts = ", ".join([json.dumps(s) for s in self.packages])
        return f'{{"filename": "{self.filename}", "decls": [{sts}, {pts}]}}'

    def run(self, G, strat="main"):
        """
        Apply rs or the last loaded one to [gr]
        :param grs_data: a graph rewriting system or a Grew string representation of a grs
        :param G: the graph, either a str (in grew format) or a dict
        :param strat: the strategy (by default "main")
        :return: the list of rewritten graphs
        """

        try:
            if self.index < 0: #not loaded
                index,_ = GRS.load_grs(self.json())
                self.index = index
            req = {
            "command": "run",
            "graph": G.json(),
            "grs_index": self.index,
            "strat": strat
            }
            print(req)
            reply = network.send_and_receive(req)
            return utils.rm_dups(reply)
        except utils.GrewError as e:
            raise utils.GrewError(
                {"function": "grew.run", "strat": strat, "message": e.value})

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
            return Graph(network.send_and_receive(req))
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
            "pattern": pattern.json(),
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
            "pattern": pattern.json(),
            }
            return network.send_and_receive(req)
        except utils.GrewError as e:
            raise utils.GrewError(
            {"function": "grew.corpus_count", "message": e.value})
    

