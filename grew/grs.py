import json
import os.path

from . import network
from . import utils
from .grew import JSON
from grew.graph import Graph
from .corpus import AbstractCorpus, Corpus

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
    def from_json(cls, json_data : JSON) :
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

    def __init__(self,args,load=False):
        """Load a grs stored in a file
        :param data: either a file name or a Grew string representation of a grs
        :or kwargs contains explicitly the parts of the grs
        :return: an integer index for latter reference to the grs
        :raise an error if the file was not correctly loaded
        """
        if isinstance(args,str):
            agrs = AbstractGRS(args)
            json_data = agrs.json()
            res = Package._from_json(json_data["decls"])
            super().__init__(res)
        elif isinstance(args, dict):
            super().__init__( args )
        if load:
            self.agrs = AbstractGRS(self)
        else:
            self.agrs = None

    def __str__(self):
        return super().__str__()

    def load(self):
        self.agrs = AbstractGRS(self)

    def run(self, G, strat="main"):
        if not self.agrs:
            raise RuntimeError("load your GRS before you run it")
        return self.agrs.run(G, strat)

class AbstractGRS:

    def __init__(self, args):
        """Load a grs stored in a file
        :param data: either a file name or a Grew string representation of a grs
        :or kwargs contains explicitly the parts of the grs
        :return: an integer index for latter reference to the grs
        :raise an error if the file was not correctly loaded
        """
        if isinstance(args, str):
            if os.path.isfile(args):
                req = {"command": "load_grs", "filename": args}
            else:
                req = {"command": "load_grs", "str": args}
        elif isinstance(args, GRS):
            req = {"command": "load_grs", "json": args.json_data()}
        else:
            raise ValueError(f"cannot build a grs with {args}")
    
        reply = network.send_and_receive(req)
        self.id = reply["index"]    

    def json(self):
        req = {"command": "json_grs", "grs_index": self.id}
        return network.send_and_receive(req)

    def __str__(self):
        return f"GRS({self.id})"

    def run(self, data, strat="main"):
        """
        run a Grs on a graph
        :param grs_data: a graph rewriting system or a Grew string representation of a grs
        :param G: the graph, either a str (in grew format) or a dict
        :param strat: the strategy (by default "main")
        :return: the list of rewritten graphs
        """
        if isinstance(data, Graph):
            req = {
                "command": "grs_run_graph",
                "graph": json.dumps(data.json_data()),
                "grs_index": self.id,
                "strat": strat
            }
            reply = network.send_and_receive(req)
            return [Graph(s) for s in reply]
        elif isinstance(data, AbstractCorpus):
            req = {
                "command": "grs_run_corpus",
                "corpus": data.get_id(),
                "grs_index": self.id,
                "strat": strat
            }
            reply = network.send_and_receive(req)
            return {sid: [Graph(s) for s in L] for sid, L in reply.items() } 
        elif isinstance(data, Corpus):
            return {sid: self.run(g) for sid,g in data.items() } 

    def apply(self, data, strat="main", abstract=True):
        """
        run a Grs on a graph or corpus
        :param grs_data: a graph rewriting system or a Grew string representation of a grs
        :param G: the graph, either a str (in grew format) or a dict
        :param strat: the strategy (by default "main")
        :return: the rewritten graph and an error if there is not exaclty one output graph
        """
        if isinstance(data, Graph):
            req = {
                "command": "grs_apply_graph",
                "graph": json.dumps(data.json_data()),
                "grs_index": self.id,
                "strat": strat
            }
            reply = network.send_and_receive(req)
            return Graph(reply)
        elif isinstance(data, AbstractCorpus):
            req = {
                "command": "grs_apply_corpus",
                "corpus": data.get_id(),
                "grs_index": self.id,
                "strat": strat
            } # return None because inplace
            network.send_and_receive(req)
            return data if abstract else Corpus (data)
        elif isinstance(data, Corpus):
            acorpus = AbstractCorpus(data)
            self.apply(acorpus, strat, abstract)
