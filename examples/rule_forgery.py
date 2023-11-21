import classifier
import numpy as np
import re

# UU$se local grew lib
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")))
from grewpy.graph import Fs_edge
from grewpy.sketch import Sketch
from grewpy import Corpus, GRS
from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, CorpusDraft, Package, Delete_feature
from sklearn.tree import export_graphviz

cpt = iter(range(1000000))

def module_name(*t):
    x = str(t)
    x = re.sub(r"\W","",x)
    return x

class WorkingGRS(GRSDraft):
    def __init__(self, *args, **kwargs):
        """
        local class: like a GRSDraft with an additional evaluation of the rules
        each rule is evaluated by two numbers: number of good application, 
        total number of application
        """
        super().__init__(*args, **kwargs)
        self.eval = {x: (0, 0) for x in self}

    def __setitem__(self, __key, __value):
        self.eval[__key] = __value[1]
        return super().__setitem__(__key, __value[0])

    def __ior__(self, other):
        super().__ior__(other)
        self.eval |= other.eval
        return self

def anomaly(obsL,  threshold):
    """
        L is a key within self
        return for L an edge and its occurrence evaluation 
        and number of total occurrences if beyond base_threshold
    """
    s = sum(obsL.values())
    for x, v in obsL.items():
        if v > threshold * s and x:
            return (x, v, s)
    return None, None, None

def build_rules(sketch, observation, param, rule_name, verbose=False):
    """
    search a rule adding an edge X -> Y, given a sketch  
    we build the clusters, then
    for each pair (X, upos=U1), (Y, upos=U2), we search for 
    some edge e occuring at least with probability base_threshold
    in which case, we define a rule R: 
    base_pattern /\ [X.upos=U1] /\ [Y.upos=U2] => add_edge X-[e]-Y
    """
    def crit_to_request(crit, val):
        if ".label" in crit:
            edge_name = re.match("(.*?).label", crit).group(1)
            clauses = Fs_edge.decompose_edge(val)
            return ";".join((f'{edge_name}.{a}="{b}"' for a, b in clauses.items()))
        return f'{crit}="{val}"'
    rules = WorkingGRS()
    loose_rules = WorkingGRS()
    for parameter in observation:
        if verbose:
            print(observation[parameter])
        x, v, s = anomaly(observation[parameter], param["base_threshold"])
        if x:
            extra_pattern = [crit_to_request(crit, val) for (
                crit, val) in zip(sketch.cluster_criterion, parameter)]
            P = Request(sketch.P, *extra_pattern)
            x0 = Fs_edge(x)  # {k:v for k,v in x}
            c = Add_edge("X", x0, "Y")
            R = Rule(P, Commands(c))
            rn = module_name(f"_{'_'.join(parameter)}_{rule_name}")
            rules[rn] = (R, (x, (v, s)))
        else:
            ...
    return rules, loose_rules

def refine_rule(R, corpus, param) -> list[Rule]:
    """
    Takes a request R, tries to find variants
    the result is the list of rules that refine pattern R
    for DEBUG, we return the decision tree classifier
    """
    res = []
    matchings = corpus.search(R)
    """
    clf = classifier.Classifier(matchings, corpus, param)
    if clf.clf:
        #branc, leaves = classifier.back_tree(clf.clf.tree_)
        #ileaves = [n for n in leaves if clf.clf.tree_.impurity[n] < param["node_impurity"]]
        branches = clf.find_classes(param)  # extract interesting branches
        #debranches = classifier.back_tree(clf.clf.tree_)
        for node in branches:
            branch = branches[node]
            request = Request(R)  # builds a new Request
            for feature_index, negative in branch:
                n, feat, feat_value = clf.fpat[feature_index]
                feat_value = feat_value.replace('"', '\\"')
                if negative:
                    request = request.without(f'{n}[{feat}="{feat_value}"]')
                else:
                    request.append("pattern", f'{n}[{feat}="{feat_value}"]')
            e = clf.y1[clf.clf.tree_.value[node].argmax()]
            if e:  # here, e == None if there is no edges X -> Y
                rule = Rule(request, Commands(Add_edge("X", e, "Y")))
                res.append(rule)
    return res, clf
    """
    return classifier.build_rules(matchings, corpus, R, param), None



def refine_rules(Rs, corpus, param, verbose=False):
    """
    as above, but applies on a list of rules
    and filter only "correct" rules, see `param`
    return the list of refined version of rules Rs
    """
    Rse = WorkingGRS()
    for rule_name in Rs.rules():
        R = Rs[rule_name]
        v, s = Rs.eval[rule_name][1]
        if v < param["min_occurrence_nb"]:
            pass
        elif v/s < param["valid_threshold"]:
            new_r, _ = refine_rule(R.request, corpus, param)
            if len(new_r) >= 1:
                if verbose:
                    print("--------------------------replace")
                    print(R)
                for r in new_r:
                    if verbose:
                        print("by : ")
                        print(r)
                    X = ",".join(f'{k}="{v}"' for k,
                                 v in r.commands[0].e.items())
                    x1 = corpus.count(Request(r.request, f"X-[{X}]->Y"))
                    x2 = corpus.count(
                        Request(r.request).without(f"X-[{X}]->Y"))
                    Rse[f"{rule_name}_enhanced{next(cpt)}"] = (
                        r, x1/(x1+x2+1e-10))
        else:
            Rse[rule_name] = (R, s)
    return Rse


"""
Learning sketches
"""
def edge_between_X_and_Y(P):
    return Request(P, 'e:X->Y')


def no_edge_between_X_and_Y(P):
    return Request(P).without('e:X-[^ANCESTOR]->Y')


def simple_sketch(r):
    return Sketch(r, ["X.upos", "Y.upos"], edge_between_X_and_Y, no_edge_between_X_and_Y, "e.label")


def apply_sketches(sketches, corpus, param):
    """
    find rules from sketches
    """
    rules = WorkingGRS()
    loose_rules = WorkingGRS()
    for sketch_name in sketches:
        sketch = sketches[sketch_name]
        r1, l1 = build_rules(sketch, sketch.cluster(
            corpus), param, sketch_name)
        rules |= r1
        loose_rules |= l1
    return rules, loose_rules


def adjacent_rules(corpus: Corpus, param) -> WorkingGRS:
    """
    build all adjacent rules. They are supposed to connect words at distance 1
    """
    sadj = dict()
    sadj["adjacent_lr"] = simple_sketch(Request("X[];Y[head];X<Y"))
    sadj["adjacent_rl"] = simple_sketch(Request("X[];Y[head];Y<X"))
    sadj["adj2_lr"] = sketch_with_parameter(Request("X[];Y[head];Z[];X<Z;Z<Y"),["Z.upos"])
    sadj["adj2_rl"] = sketch_with_parameter(Request("X[];Y[head];Z[];Y<Z;Z<X"),["Z.upos"])
    sadj["no_intermediate_1"] = simple_sketch(
        Request("X[];Y[head];X<<Y").without("Z[];X<<Z;Z<<Y;X.upos=Z.upos"))
    sadj["no_intermediate_2"] = simple_sketch(
        Request("X[];Y[head];X<<Y").without("Z[];X<<Z;Z<<Y;Y.upos=Z.upos"))
    sadj["no_intermediate_3"] = simple_sketch(
        Request("X[];Y[head];Y<<X").without("Z[];Y<<Z;Z<<X;Y.upos=Z.upos"))
    sadj["no_intermediate_4"] = simple_sketch(
        Request("X[];Y[head];Y<<X").without("Z[];Y<<Z;Z<<X;X.upos=Z.upos"))

    nodes = ['f:X -> Z', 'f:Y -> Z', 'f:Z->X', 'f:Z->Y']
    ordres = ['X<Y', 'X>Y', 'Z<Y', 'Z>Y', 'X<Z',
              'X>Z', 'Z<<Y', 'Z>>Y', 'X<<Z', 'X>>Z']
    on_label = [("Z.upos",), ("f.label",), tuple()]
    for ns in nodes:
        for o in ordres:
            for extra in on_label:
                sadj[module_name((ns, o, extra, next(cpt)))] =\
                    Sketch(Request('X[];Y[head]', ns, o), ["X.upos", "Y.upos"] + list(extra),
                           edge_between_X_and_Y,
                           no_edge_between_X_and_Y, "e.label")
    return apply_sketches(sadj, corpus, param)


def sketch_with_parameter(req, extra_labels=[]):
    return Sketch(Request(req), ["X.upos", "Y.upos"] + extra_labels, edge_between_X_and_Y, no_edge_between_X_and_Y, "e.label")


def local_rules(corpus: Corpus, param) -> WorkingGRS:
    local = ["h:U$->V$;U$<<X;U$<<Y;X<<V$;Y<<V$", "h:V$->U$;U$<<X;U$<<Y;X<<V$;Y<<V$"]
    sadj = dict()
    for loc in local:
        sadj[module_name("loc_lr", loc)] = sketch_with_parameter(
            Request(loc, "X[];Y[head];X<Y"))
        sadj[module_name("loc_rl", loc)] = sketch_with_parameter(
            Request(loc, "X[];Y[head];Y<X"))
        sadj[module_name("int_1", loc)] = sketch_with_parameter(Request(
            loc, "X[];Y[head];X<<Y").without("Z[];X<<Z;Z<<Y;X.upos=Z.upos"))
        sadj[module_name("int_2", loc)] = sketch_with_parameter(Request(
            loc, "X[];Y[head];X<<Y").without("Z[];X<<Z;Z<<Y;Y.upos=Z.upos"))
        sadj[module_name("int_3", loc)] = sketch_with_parameter(
            Request(loc, "X[];Y[head];Y<<X").without("Z[];Y<<Z;Z<<X;Y.upos=Z.upos"))
        sadj[module_name("int_4", loc)] = sketch_with_parameter(
            Request(loc, "X[];Y[head];Y<<X").without("Z[];Y<<Z;Z<<X;X.upos=Z.upos"))

    nodes = ['f:X -> Z$', 'f:Y -> Z$', 'f:Z$->X', 'f:Z$->Y']
    ordres = ['X<Y', 'X>Y', 'X<<Y','Y<<X']
    on_label = [("f.label",), tuple()]
    for loc in local:
        for ns in nodes:
            for o in ordres:
                for extra in on_label:
                    sadj[module_name((loc, ns, o, extra, next(cpt)))] =\
                Sketch(Request(loc,'X[];Y[head];e<>h', ns, o), ["X.upos", "Y.upos", "Z$.upos"] + list(extra), 
                edge_between_X_and_Y, 
                no_edge_between_X_and_Y, "e.label")
    return apply_sketches(sadj, corpus, param)

def feature_value_occurences(matchings, corpus, skipped_features, max_per_feature):
    """
    given a matchings corresponding to some request on the corpus,
    return a dict mapping (n,feature) =>(values)=>occurrences to its occurence number in matchings
    within the corpus. n : node name, feature like 'Gender', values like 'Fem'
    """
    observation = dict()
    for m in matchings:
        graph = corpus[m["sent_id"]]
        nodes = m['matching']['nodes']
        for n in nodes:
            N = graph[nodes[n]]  # feature structure of N
            for k, v in N.items():
                if k not in skipped_features:
                    if (n, k) not in observation:
                        observation[(n, k)] = dict()
                    observation[(n, k)][v] = observation[(n, k)].get(v, 0)+1
    obs = dict()
    for (n,k) in observation:
        if len(observation[(n,k)]) < 20:
            for v, o in observation[(n,k)].items():
                obs[(n,k,v)] = o
        else:
            L = [(o,v) for v,o in observation[(n,k)].items()]
            L.sort(reverse=True)
            for (o,v) in L[:max_per_feature]:
                obs[(n,k,v)] = o
    return obs