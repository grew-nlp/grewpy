"""
Grew module : anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
from operator import ne
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

cpt = iter(range(2**30))#an iterator for implicit names

def init(dev=False):
    """
    Initialize connection to GREW library
    :return: the ouput of the subprocess.Popen command.
    """
    return network.init(dev)
init()

class NamedList():
    """
    a list of things with an underlying name:
    things may be json()'ed
    """
    def __init__(self,sort,*L):
        """
        L is a list of
         - ";" separated clause or
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

    def json(self):
        t = [x.json() if "json" in dir(x) else x for x in self.items]
        return f"{self.sort}{{{';'.join(t)}}}"
    def __str__(self):
        return f"{self.sort}{{{';'.join(self.items)}}}"

class Pattern():
    def __init__(self, *L):
        """
        L is a list of ClauseList or pairs (sort,clauses)
        """
        self.items = tuple(C if isinstance(C,NamedList) else NamedList(*C)
                        for C in L)

    def json(self):
        return "".join([C.json() for C in self.items])

    def __getitem__(self,i):
        return self.items[i]

    def __str__(self):
        return "\n".join([str(x) for x in self.items])

class Strategy():
    def __init__(self,name, data):
        self.data = data
        self.name = name
    def json(self):
        return f'{{"type":"strat", "id": "{self.name}", "data" : "{self.data}"}}'
    def __str__(self):
        return f"strat {self.name} {{{self.data}}}"

class Command():
    def __init__(self, *L):
        self.items = []
        for elt in L:
            if isinstance(elt,str):
                self.items += [t.strip() for t in elt.split(";") if t.strip()]
            elif isinstance(elt,list):
                self.items += elt

    def __str__(self):
        c = ";".join(self.items)
        return f"commands {{{c}}}"

class Rule():
    def __init__(self, name, pattern, cmds):
        self.pattern = pattern
        self.commands = cmds
        self.name = name
    def json(self):
        p = self.pattern.json()
        c = self.commands.json()
        return f"rule {self.name} {{ {p}\n {c} }}"
    def __str__(self):
        return f"rule {self.name}{{{self.pattern}\n{self.commands}}}"

class GRS():

    def load_grs(data):
        """load data (either a filename or a json encoded string) within grew"""
        name = next(cpt)
        if not os.path.isfile(data):
            f = tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".grs")
            f.write(data)
            f.flush() # The file can be empty if we do not flush
            req = {"command": "load_grs", "filename": f.name}
            reply = network.send_request(req)
            name = data
        else:
            req = {"command": "load_grs", "filename": data}
            reply = network.send_request(req)
        index = reply["index"]
        return index,name

    def __ii(self,n,p,s,r,index=-1):
        self.name =n
        self.packages = p
        self.rules = r
        self.strats = s
        self.index = index

    def __init__(self, data=None, **kwargs):
        """Load a grs stored in a file
        :param data: either a file name or a Grew string representation of a grs
        :return: an integer index for latter reference to the grs
        :raise an error if the file was not correctly loaded
        """
        if len(kwargs) > 0:
            #explicit declaration
            GRS.__ii(self,data if data else next(cpt), 
            kwargs.get('packages', list()), 
            kwargs.get('strats', list()),
            kwargs.get('rules', list()))
        elif data is None:
            GRS.__ii(self,'',list(),list(),list())
            return
        elif isinstance(data, str):
            index,name = GRS.load_grs(data)
            GRS.__ii(self,str(name),list(),list(),list(),index)
            req = {"command": "json_grs", "grs_index": index}
            json_data = network.send_request(req)
            print ("--------------------")
            print (json.dumps(json_data, indent=2))
            print ("--------------------")
            for k,v in json_data["decls"].items():
                if isinstance(v, str): ## strat
                    pass
                    #TO BE IMPLEMENTED
                    # self.strats.append(Strategy(d['strat_name'], d['strat_def']))
                elif 'commands' in v: ## rule
                    pass
                    #self.rules.append(Rule(d['rule_name'],d['rule'],''))
                    #TO BE IMPLEMENTED
                elif 'decls' in v: ## package
                    pass
                    #TO BE IMPLEMENTED
                else:
                    raise utils.GrewError(f"{v} is not a valid decl")
        else:
            pass
            """
            TO BE IMPLEMENTED
            """

    # def json(self):
    #     sts = ", ".join(
    #         [f"{{'strat_name' : {s}, 'strat_def':{v}}}" for s, v in self.strats.items()])
    #     pts = ", ".join([json.dumps(s) for s in self.packages])
    #     return f'{{"filename": "{self.filename}", "decls": [{sts}, {pts}]}}'

    def json(self):
        return f'{{"name": "{self.name}", "strats": {self.strats}}}'

    def __str__(self):
        """
        a string representation of self
        """
        sts = "\n".join([f"{s}" for s in self.strats])
        rls = "\n".join([f"{r}" for r in self.rules])
        return sts+"\n"+rls


    def run(self, G, strat="main"):
        """
        Apply rs or the last loaded one to [gr]
        :param grs_data: a graph rewriting system or a Grew string representation of a grs
        :param G: the graph, either a str (in grew format) or a dict
        :param strat: the strategy (by default "main")
        :return: the list of rewritten graphs
        """
        if self.index < 0: #not loaded
            index,_ = GRS.load_grs(str(self))
            self.index = index
        req = {
            "command": "run",
            "graph": G.json(),
            "grs_index": self.index,
            "strat": strat
        }
        reply = network.send_request(req)
        return [Graph(s) for s in reply]

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
                reply = network.send_request(req)
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

    def search(self,pattern):
        """
        Search for [pattern] into [corpus_index]
        :param patten: a string pattern
        :param corpus_index: an integer given by the [corpus] function
        :return: the list of matching of [pattern] into the corpus
        """
        return network.send_request({
            "command": "corpus_search",
            "corpus_index": self.id,
            "pattern": pattern.json(),
            })

    def count(self,pattern):
        """
        Count for [pattern] into [corpus_index]
        :param patten: a string pattern
        :param corpus_index: an integer given by the [corpus] function
        :return: the number of matching of [pattern] into the corpus
        """
        return network.send_request({
            "command": "corpus_count",
            "corpus_index": self.id,
            "pattern": pattern.json(),
            })
