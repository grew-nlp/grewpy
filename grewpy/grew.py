"""
Grew module : anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
from multiprocessing.sharedctypes import Value
import os.path
import tempfile
import json

import network
from graph import Graph
from utils import GrewError

''' Library tools '''

def init(dev=False):
    """
    Initialize connection to GREW library
    :return: the ouput of the subprocess.Popen command.
    """
    return network.init(dev)
init()

def pattern(s):
    return ("pattern", s)
def without(s):
    return ("without",s)

class ClauseList():
    def __init__(self,sort : str,*L):
        """
        sort in {"without", "pattern", "global"}
        L is a list of
         - ";" separated clauses or
         - a list of items
         - they will be concatenated
        """
        self.sort = sort 
        self.items = tuple()
        for elt in L:
            if isinstance(elt,str):
                self.items += tuple(c.strip() for c in elt.split(";") if c.strip())
            else:
                self.items += tuple(elt)

    def json_data(self):
        return {self.sort : list(self.items)}
    
    @classmethod
    def from_json(cls, json_data):
        k = list(json_data)[0]
        v = json_data[k]
        return cls(k,*v)

    def __str__(self):
        its = ";".join([str(x) for x in self.items])
        return f"{self.sort} {{{its}}}"

class Request():
    """
    lists of ClauseList
    """
    def __init__(self, *L):
        """
        L is either a list of 
         - ClauseList or
         - (pattern) string or a 
         - Request (for copies)
        """
        elts = tuple()
        for e in L:
            if isinstance(e,str):
                elts += (ClauseList("pattern", e),)
            elif isinstance(e,ClauseList):
                elts += (e,)
            elif isinstance(e,Request):
                elts += e.items
            else:
                raise ValueError(f"{e} cannot be used to build a Request")
        self.items = elts

    def without(self, *L):
        self.items += tuple(ClauseList("without", e) for e in L)
        return self   
        
    @classmethod
    def from_json(cls,json_data):
        elts = [ClauseList.from_json(c) for c in json_data]
        return cls(*elts)

    def json_data(self):
        return [x.json_data() for x in self.items]
    
    def __str__(self):
        return "\n".join([str(e) for e in self.items])

class Command(list):
    def __init__(self, *L):
        super().__init__()
        for elt in L:
            if isinstance(elt,str):
                self += [t.strip() for t in elt.split(";") if t.strip()]
            elif isinstance(elt,list):
                self += elt

    def __str__(self):
        c = ";".join(self)
        return f"commands {{{c}}}"

    @classmethod
    def from_json(cls, json_data):
        return cls(*json_data)

class Rule():
    def __init__(self, request : Request, cmd_list : Command):
        self.request = request
        self.commands = cmd_list

    def json_data(self):
        p = self.request.json_data()
        return {"request" : p, "commands" : self.commands}

    def __str__(self):
        return f"{str(self.request)}\n{str(self.commands)}"
    
    @classmethod
    def from_json(cls,json_data):
        # print(json_data)
        reqs = Request.from_json(json_data["request"])
        cmds = Command.from_json(json_data["commands"])
        return cls(reqs,cmds) 

class Package(dict):
    """
    dict mapping names to rule/package/strategies"""

    @classmethod
    def from_json(cls, json_data):
        res = Package._from_json(json_data)
        return cls(res)

    def _from_json(json_data):
        res = dict()
        for k,v in json_data.items():
            if isinstance(v,str):
                res[k] = v
            elif "decls" in v: #it is a package
                res[k] = Package.from_json(v["decls"])
            else:
                res[k] = Rule.from_json(v)
        return res

    def json_data(self):
        elts = dict()
        for k,v in self.items():
            elts[k] = v if isinstance(v,str) else v.json_data()
        return {"decls" : elts}

    def __str__(self):
        res = [f"strat {k} {{{self[k]}}}" for k in self.strategies()] +\
            [f"package {k} {{{str(self[k])}}}" for k in self.packages()] +\
            [f"rule {k} {{{str(self[k])}}}" for k in self.rules()]
        return "\n".join(res)

    def rules(self):
        return filter(lambda x: isinstance(self[x], Rule), self.__iter__())

    def packages(self):
        return filter(lambda x: isinstance(self[x], Package), self.__iter__())

    def strategies(self):
        return filter(lambda x: isinstance(self[x], str), self.__iter__())


class GRS(Package):

    def _load(data):
        """load data (either a filename or a json encoded string) within grew"""
        if os.path.isfile(data):
            req = {"command": "load_grs", "filename": data}
        else:
            req = {"command": "load_grs", "str": data}
        reply = network.send_request(req)
        index = reply["index"]
        return index

    def _build(self):
        """ ensure that the GRS is loaded (and call Ocaml if needed)"""
        if not self.index:
            data = self.json_data()
            req = {"command": "load_grs", "json": data}
            reply = network.send_request(req)
            index = reply["index"]
            self.index = index

    def __init__(self,args):
        """Load a grs stored in a file
        :param data: either a file name or a Grew string representation of a grs
        :or kwargs contains explicitly the parts of the grs
        :return: an integer index for latter reference to the grs
        :raise an error if the file was not correctly loaded
        """
        if isinstance(args,str):
            index = GRS._load(args)
            req = {"command": "json_grs", "grs_index": index}
            json_data = network.send_request(req)
            res = Package._from_json(json_data["decls"])
            super().__init__(res)
            self.index = index
        elif isinstance(args, dict):        
            super().__init__( args )
            self.index = 0
        
    def __str__(self):
        return super().__str__()

    def run(self, G, strat="main"):
        """
        Apply rs or the last loaded one to [gr]
        :param grs_data: a graph rewriting system or a Grew string representation of a grs
        :param G: the graph, either a str (in grew format) or a dict
        :param strat: the strategy (by default "main")
        :return: the list of rewritten graphs
        """
        self._build() # first ensure that the GRS is loaded in Ocaml
        req = {
            "command": "run",
            "graph": G.json(),
            "grs_index": self.index,
            "strat": strat
        }
        # print("---------------------")
        # print(req)
        # print("-------------------------")
        reply = network.send_request(req)
        return [Graph(s) for s in reply]


    def __setitem__(self,x,v):
        self.index = 0
        super().__setitem__(x,v)


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


    def search(self,request):
        """
        Search for [pattern] into [corpus_index]
        :param patten: a string pattern
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [pattern] into the corpus
        """
        return network.send_request({
            "command": "corpus_search",
            "corpus_index": self.id,
            "request": request.json_data(),
            })

    def count(self,request):
        """
        Count for [pattern] into [corpus_index]
        :param patten: a string pattern
        :param corpus_index: an integer given by the [corpus] function
        :return: the number of matching of [pattern] into the corpus
        """
        return network.send_request({
            "command": "corpus_count",
            "corpus_index": self.id,
            "pattern": request.json_data(),
            })
