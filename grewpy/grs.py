import json
import os.path

from . import network
from . import utils
from .grew import JSON
from grewpy.graph import Graph
from .corpus import Corpus, CorpusDraft, GrewError

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

    def __repr__(self):
        return f"{self.sort} {{{ ';'.join([str(x) for x in self.items]) }}}"

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
            elif isinstance(e,tuple): #supposed to be a list of ClauseList
                elts += e
            else:
                try:
                    #suppose it is a generator
                    for x in e:
                        elts += (x,)
                except:
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

    def __iter__(self):
        return iter(self.items)

    def pattern(self):
        """
        return the pattern of self as a tuple
        """
        return Request(p for p in self.items if p.sort == "pattern")

    def append(self, *L):
        """
        Append a new ClauseList to the Request
        L is given either as a pair (s,t) with "s \in {'pattern','without','meta'} and t : str 
        or L[0] is a ClauseList 
        """
        if len(L) == 2:
            self.items = self.items + (ClauseList(L[0], L[1]),)
        elif len(L) == 1:
            self.items = self.items + L
        else:
            raise ValueError(f"cannot build a clause list with {L}")


class Command:
    def __init__(self,s):
        """
        self.item = str representation of the command
        """
        self.item = s

    def json_data(self):
        return self.item
    
    def __str__(self):
        return self.item

    def safe(self):
        """
        return a clause list for a safe request
        """
        raise NotImplementedError ("not yet implemented")

class Add_edge(Command):
    def __init__(self,X,e,Y):
        super().__init__(f"add_edge {X}-[{e}]->{Y}")
        self.X, self.e, self.Y = X, e, Y

    def safe(self):           
        return ClauseList("without",f"{self.X} -[{self.e}]->{self.Y}")

class Delete_edge(Command):
    def __init__(self, X, e, Y):
        super().__init__(f"del_edge {X}-[{e}]->{Y}")
        self.X, self.e, self.Y = X, e, Y

    def safe(self):           
        return ClauseList("pattern", f"{self.X} -[{self.e}]->{self.Y}")


class Commands(list):
    def __init__(self, *L):
        super().__init__()
        for elt in L:
            if isinstance(elt,str):
                self += [t.strip() for t in elt.split(";") if t.strip()]
            elif isinstance(elt,list):
                self += elt
            elif isinstance(elt, Command):
                self.append(elt)

    def __str__(self):
        c = ";".join([str(x) for x in self])
        return f"commands {{{c}}}"

    @classmethod
    def from_json(cls, json_data):
        return cls(*json_data)
    
    def json_data(self):
        return [x if isinstance(x,str) else x.json_data() for x in self]

class Rule():
    def __init__(self, request : Request, cmd_list : Commands, lexicons =None):
        self.request = request
        self.commands = cmd_list
        self.lexicons = lexicons if lexicons else dict()

    def json_data(self):
        p = self.request.json_data()
        c = self.commands.json_data()
        return {"request" : p, "commands" : c, "lexicons" : json.dumps(self.lexicons)}

    def __str__(self):
        return f"{str(self.request)}\n{str(self.commands)}"

    def __repr__(self):
        return f"{str(self.request)}\n{str(self.commands)}"

    @classmethod
    def from_json(cls,json_data):
        # print(json_data)
        reqs = Request.from_json(json_data["request"])
        cmds = Commands.from_json(json_data["commands"])
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


class GRSDraft(Package):
    """
    A GRSDraft is a structure that gives access to the internals of a 
    Graph Rewriting System: packages, rules, patterns, strategies, etc
    It cannot be used to perform rewriting, for that, use a GRS
    """

    def __init__(self,args):
        """Load a grs stored in a file
        :param data: either a file name or a Grew string representation of a grs
        :or kwargs contains explicitly the parts of the grs
        :return: an integer index for latter reference to the grs
        :raise an error if the file was not correctly loaded
        """
        if isinstance(args,str):
            agrs = GRS(args)
            json_data = agrs.json()
            res = Package._from_json(json_data["decls"])
            super().__init__(res)
        elif isinstance(args, dict):
            super().__init__( args )

    def __str__(self):
        return super().__str__()

class GRS:
    """
    An abstract GRS. Offers the possibility to apply rewriting.
    The object is abstract and cannot be changed. 
    For that, use a GRSDraft
    """

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
        elif isinstance(args, GRSDraft):
            req = {"command": "load_grs", "json": args.json_data()}
        elif isinstance(args, dict):
            """
            suppose it is a GRS style
            """
            try:
                grs = GRSDraft(args)
                req = {"command": "load_grs", "json": grs.json_data()}
            except GrewError as e:
                raise ValueError(f"cannot build a grs with {args}\n {e.message}")
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
        elif isinstance(data, Corpus):
            req = {
                "command": "grs_run_corpus",
                "corpus": data.get_id(),
                "grs_index": self.id,
                "strat": strat
            }
            reply = network.send_and_receive(req)
            return {sid: [Graph(s) for s in L] for sid, L in reply.items() } 
        elif isinstance(data, CorpusDraft):
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
        elif isinstance(data, Corpus):
            req = {
                "command": "grs_apply_corpus",
                "corpus": data.get_id(),
                "grs_index": self.id,
                "strat": strat
            } # return None because inplace
            network.send_and_receive(req)
            return data if abstract else CorpusDraft (data)
        elif isinstance(data, CorpusDraft):
            acorpus = Corpus(data)
            self.apply(acorpus, strat, abstract)
