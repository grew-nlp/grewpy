import json
import warnings
import sys
import os.path
from typing import List, Tuple

from . import network
from .grew import JSON
from grewpy.graph import Graph
from .corpus import Corpus, CorpusDraft, GrewError

import lark
import sys

request_grammar = """
%import common.ESCAPED_STRING
%import common.WS
COMMENT: /%[^\n]*/x
%ignore COMMENT
%ignore WS
SYMBOLS.2 : "-"|"]"|"["|/[\/*!<>;,_=.:#@$|^()]/
TOKEN : (/\\w/|SYMBOLS)+
lines : (TOKEN|ESCAPED_STRING)*
KEYWORDS : "pattern" | "global" | "with" | "without"
request_item : KEYWORDS "{" lines "}"
request : request_item*
"""
req_grammar_ = lark.Lark(request_grammar, start="request")

class RequestItem():
    def __init__(self,sort : str,*L):
        """
        sort in {"global", "pattern", "without", "with"}
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
        return cls(k,v)

    def __str__(self):
        its = ";".join([str(x) for x in self.items])
        return f"{self.sort} {{{its}}}"

    def __repr__(self):
        return f"{self.sort} {{{ ';'.join([str(x) for x in self.items]) }}}"

class Request():
    '''
    lists of ClauseList
    '''
    def __init__(self, *L):
        '''
        L is either:
         - nothing,
         - an other request (for a copy)
         - or a grew-syntax request string
         - or a list of requestitems
        '''
        if len(L) == 0:
            self.items = tuple()
            return
        if len(L) == 1:
            R = L[0]
            if isinstance(R, str):
                self.items = tuple(RequestItem(t,data) for t,data in Request.parse_request(R))
                return
            if isinstance(R, Request):
                self.items = tuple(R.items)
                return
            if isinstance(R,tuple):
                self.items = R
                return
        if all (isinstance(elt, RequestItem) for elt in L):
            self.items = tuple(L)
            return
        raise TypeError(f"cannot build a request out of {L}")

    def without(self, *L):
        return Request (self.items + tuple(RequestItem("without", e) for e in L))

    def with_(self, *L):
        return Request (self.items + tuple(RequestItem("with", e) for e in L))

    def global_(self, *L):
        return Request (self.items + tuple(RequestItem("global", e) for e in L))

    def pattern(self, *L):
        return Request (self.items + tuple(RequestItem("pattern", e) for e in L))

    @classmethod
    def from_json(cls,json_data):
        if isinstance(json_data,str):
            return cls.parse(json_data)
        if len(json_data) > 0 and isinstance(json_data[0], str):
            return cls.parse("\n".join(json_data))
        else:
            elts = [RequestItem.from_json(c) for c in json_data]
            return cls(*elts)

    @staticmethod
    def parse_request(s : str) -> List[Tuple[str,str]]:
        try:
            p = req_grammar_.parse(s)
            items = []
            for N in p.children:
                content = "".join([M.value for M in N.children[1].children])
                items.append( (N.children[0].value, content))
            return items
        except Exception as e:
            print(f"Could not parse: {e}")

    def json_data(self):
        return [x.json_data() for x in self.items]

    def __str__(self):
        return "\n".join([str(e) for e in self.items])

    def __iter__(self):
        return iter(self.items)

    def append(self, *L):
        """
        Append a new RequestItem to the Request
        L is given either as a pair (s,t) with "s \\in {'pattern','without','meta'} and t : str
        or L[0] is a RequestItem
        """
        if len(L) == 2:
            self.items = self.items + (RequestItem(L[0], L[1]),)
        elif len(L) == 1 and isinstance(L[0],RequestItem):
            self.items = self.items + L
        else:
            raise ValueError(f"cannot build a request with {L}")

    def named_entities(self):
        req = {"command": "request_named_entities", "request": self.json_data(),}
        reply = network.send_and_receive(req)
        return reply

class Command:
    def __init__(self,s):
        """
        self.item = str representation of the command
        """
        self.item = s

    def json_data(self):
        return self.item

    def __str__(self):
        return str(self.item)

    def safe(self):
        """
        return a clause list for a safe request
        """
        raise NotImplementedError ("not yet implemented")

class AddEdge(Command):
    def __init__(self,X,e,Y):
        if isinstance(e, dict):
            s = ",".join(f"{k}={v}" for k,v in e.items())
        else:
            s = str(e)
        super().__init__(f"add_edge {X}-[{s}]->{Y}")
        self.X, self.e, self.Y = X, e, Y

    def safe(self):
        return RequestItem("without",self.item.replace("add_edge",""))

    def __repr__(self):
        return str(self)


class Add_edge(AddEdge):
   def __init__(self, X, f):
        warnings.warn(
            """Add_edge is deprecated and will be removed in a future version.
            Please use AddEdge instead.
            See https://grew.fr/grewpy/upgrade_0.6/
            """,
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(X, f)



class DeleteEdge(Command):
    def __init__(self, X, e, Y):
        super().__init__(f"del_edge {X}-[{e}]->{Y}")
        self.X, self.e, self.Y = X, e, Y

    def safe(self):
        return RequestItem("pattern", self.item.replace("del_edge", ""))

class Delete_edge(DeleteEdge):
   def __init__(self, X, f):
        warnings.warn(
            """Delete_edge is deprecated and will be removed in a future version.
            Please use DeleteEdge instead.
            See https://grew.fr/grewpy/upgrade_0.6/
            """,
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(X, f)


class DeleteFeature(Command):
    def __init__(self, X, f):
        super().__init__(f"del_feat {X}.{f}")
        self.X = X
        self.f = f

class Delete_feature(DeleteFeature):
   def __init__(self, X, f):
        warnings.warn(
            """Delete_feature is deprecated and will be removed in a future version.
            Please use DeleteFeature instead.
            See https://grew.fr/grewpy/upgrade_0.6/
            """,
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(X, f)
        super().__init__(X, f)

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

    def __init__(self,args=None):
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
        elif args == None:
            super().__init__()

    def __str__(self):
        return super().__str__()

    def rules(self):
        return (rule_name for rule_name in self if isinstance(self[rule_name],Rule))

    def safe_rules(self):
        """
        create a new grs with application of safe to each rule.
        self.rules() are supposed to contain only Commands of length 1 that support safe method
        """
        grs = GRSDraft()
        for rule_name in self.rules():
            rule = self[rule_name]
            cde = rule.commands[0]
            safe_request = Request(rule.request)
            safe_request.append(cde.safe())
            safe_rule = Rule(safe_request, rule.commands, rule.lexicons)
            grs[rule_name] = safe_rule
        return grs

    def onf(self, strat_name="main"):
        self[strat_name] = f'Onf(Alt({",".join(self.rules())}))'
        return self

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(str(self))

def constant_UD2bUD(cls):
    cls.UD2bUD = cls("""
package UD2bUD {
  rule enh { % remove enhanced relations
    pattern { e:N -[enhanced=yes]-> M }
    commands { del_edge e}
  }

  rule empty { % remove empty nodes
    pattern { N [wordform=__EMPTY__, textform=_] }
    commands { del_node N }
  }
}

strat main { Onf(UD2bUD) }
""")
    return cls

@constant_UD2bUD


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
                req = {"command": "load_grs", "file": args}
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
        :param data: a graph or an AbstractCorpus
        :param strat: the strategy (by default "main")
        :return: a dictionary mapping sid to the list of rewritten graphs
        """
        if isinstance(data, Graph):
            req = {
                "command": "grs_run_graph",
                "graph": json.dumps(data.json_data()),
                "grs_index": self.id,
                "strat": strat
            }
            reply = network.send_and_receive(req)
            return [Graph.from_json(s) for s in reply]
        elif isinstance(data, Corpus):
            req = {
                "command": "grs_run_corpus",
                "corpus_index": data.get_id(),
                "grs_index": self.id,
                "strat": strat
            }
            reply = network.send_and_receive(req)
            return {sid: [Graph(s) for s in L] for sid, L in reply.items() }
        elif isinstance(data, CorpusDraft):
            return {sid: self.run(g) for sid,g in data.items() }
        else:
            raise TypeError(f"GRS method 'run' cannot bu used with {data}")

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
                "corpus_index": data.get_id(),
                "grs_index": self.id,
                "strat": strat
            } # return None because inplace
            network.send_and_receive(req)
            return data if abstract else CorpusDraft (data)
        elif isinstance(data, CorpusDraft):
            corpus = Corpus(data)
            self.apply(corpus, strat, abstract)
        else:
            raise TypeError(f"GRS method 'apply' cannot bu used with {data}")
