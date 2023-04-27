from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import math
import numpy as np
import re
import argparse
import pickle

# Use local grew lib
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, CorpusDraft, Package, Delete_feature
from grewpy import Corpus, GRS, set_config
from grewpy.sketch import Sketch
from grewpy import grew_web
from grewpy.graph import Fs_edge
import classifier

cpt = iter(range(1000000))

def module_name(t):
    x = str(t)
    x = re.sub(r"['\(\),+<> \:.@]","",x)
    return x

def edge_is(e,f):
    """
    tells whether e is an f
    """
    for k in e:
        if k not in f:
            return False
        if e[k] not in f[k]:
            return False
    return True

def corpus_diff(c1,c2):
    c,l,r = 0,0,0
    for sid in c1:
        men = {m: (e, n) for n, e, m in c1[sid].triples()}
        menp = {m : (e,n) for n,e, m in c2[sid].triples()}
        S1 = set(men.keys())
        S2 = set(menp.keys())
        l += len(S1 - S2)
        r += len(S2 - S1)
        for m in S1 & S2:
            e,n = men[m]
            ep,np = menp[m]
            if n != np or not edge_is(ep,e):
                l,r = l+1,r+1
            else:
                c += 1
    precision = c / (c + l+1e-10)
    recall = c / (c + r+1e-10)
    f_measure = 2*precision*recall / (precision+recall+1e-10)
    return {
        "common": c,
        "left": l,
        "right": r,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f_measure": round(f_measure, 3),
    }



class WorkingGRS(GRSDraft):
    def __init__(self, *args, **kwargs):
        """
        local class: like a GRSDraft with an additional evaluation of the rules
        each rule is evaluated by two numbers: number of good application, 
        total number of application
        """
        super().__init__(*args, **kwargs)
        self.eval = {x: (0,0) for x in self}

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

def best_two(obsL, threshold):
    ...

def build_rules(sketch, observation, param, rule_name, rank_level=0):
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
        #x, v, s = observation.anomaly(parameter, param["base_threshold"])
        x, v, s = anomaly(observation[parameter], param["base_threshold"])
        if x:
            extra_pattern = [crit_to_request(crit, val) for (
                crit, val) in zip(sketch.cluster_criterion, parameter)]
            P = Request(sketch.P, *extra_pattern)
            x0 = Fs_edge(x[0])  # {k:v for k,v in x}
            c = Add_edge("X", x0, "Y")
            R = Rule(P,Commands(c))
            rn = module_name(f"_{'_'.join(parameter)}_{rule_name}")
            rules[rn] = (R, (x, (v, s)))
        else:
            ...
            """
            search for a loose rule
            
            vals = observation[parameter].values()
            s = sum(vals)
            T = sorted(observation[parameter].items(), key= lambda x: x[1], reverse=True)
            if T[0][0] != '' and T[1][0] != '' and (T[0][1]+T[1][1]) > 9*s/10:
                extra_pattern = [crit_to_request(crit, val) for (
                    crit, val) in zip(sketch.cluster_criterion, parameter)]
                P = Request(sketch.P, *extra_pattern)
                x0 = Fs_edge(f"{T[0][0]}|{T[1][0]}")  # {k:v for k,v in x}
                c = Add_edge("X", x0, "Y")
                R = Rule(P, Commands(c))
                rn = module_name(f"_{'_'.join(parameter)}_{rule_name}")
                loose_rules[rn] = (R, (x, (v, s)))
                """

    return rules, loose_rules


def refine_rule(R, corpus, param, rank) -> list[Rule]:
    """
    Takes a request R, tries to find variants

    the result is the list of rules that refine pattern R
    for DEBUG, we return the decision tree classifier
    """
    res = []
    matchings = corpus.search(R)
    clf = classifier.Classifier(matchings, corpus, param)
    if clf.clf:
        branches = clf.find_classes(param)  # extract interesting branches
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


def refine_rules(Rs, corpus, param, rank, debug=False):
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
            new_r, clf = refine_rule(R.request, corpus, param, rank)
            if len(new_r) >= 1:
                if debug:
                    print("--------------------------replace")
                    print(R)
                for r in new_r:
                    if debug:
                        print("by : ")
                        print(r)
                    X = ",".join(f'{k}="{v}"' for k,v in r.commands[0].e.items()) 
                    x1 = corpus.count(Request(r.request, f"X-[{X}]->Y"))
                    x2 = corpus.count(Request(r.request).without(f"X-[{X}]->Y"))
                    Rse[f"{rule_name}_enhanced{next(cpt)}"] = (r, x1/(x1+x2+1e-10))
        else:
            Rse[rule_name] = (R,s)
    return Rse

def clear_but_working(g):
    """
    delete non working edges within g
    """
    g.sucs = {n : [] for n in g.sucs}
    return (g)


"""
Operations on computations
"""

def get_best_solution(corpus_gold, corpus_start, grs) -> CorpusDraft:
    """
    grs is a GRSDraft
    return the best solution using the grs with respect to the gold corpus
    the grs does not need to be confluent. We take the best solution with 
    respect to the f-score
    """
    def f_score(t):
        return t[0]/math.sqrt((t[0]+t[1])*(t[0]*t[2])+1e-20)

    corpus = CorpusDraft(corpus_start)
    print(len(corpus_gold))
    i = 0
    for sid in corpus_gold:
        if i % (len(corpus_gold)//1) == 0:
            print(i)
        i += 1
        gs = grs.run(corpus_start[sid], 'main')
        best_fscore = 0
        for g in gs:
            fs = f_score(g.edge_diff_up_to(corpus_gold[sid]))
            if fs > best_fscore:
                best_fscore = fs
                corpus[sid] = g
    return corpus

"""
Learning sketches
"""

def edge_between_X_and_Y(P):
    return Request(P, 'e:X->Y')

def no_edge_between_X_and_Y(P):
    return Request(P).without('e:X-[^LEFT_SPAN|RIGHT_SPAN|ANCESTOR]->Y')

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
        r1,l1 = build_rules(sketch, sketch.cluster(corpus), param, sketch_name)
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
    sadj["no_intermediate_1"] = simple_sketch(Request("X[];Y[head];X<<Y").without("Z[];X<<Z;Z<<Y;X.upos=Z.upos"))
    sadj["no_intermediate_2"] = simple_sketch(Request("X[];Y[head];X<<Y").without("Z[];X<<Z;Z<<Y;Y.upos=Z.upos"))
    sadj["no_intermediate_3"] = simple_sketch(Request("X[];Y[head];Y<<X").without("Z[];Y<<Z;Z<<X;Y.upos=Z.upos")) 
    sadj["no_intermediate_4"]=simple_sketch(Request("X[];Y[head];Y<<X").without("Z[];Y<<Z;Z<<X;X.upos=Z.upos"))
    
    nodes = ['f:X -> Z', 'f:Y -> Z', 'f:Z->X', 'f:Z->Y']
    ordres = ['X<Y', 'X>Y', 'Z<Y', 'Z>Y', 'X<Z', 'X>Z', 'Z<<Y', 'Z>>Y', 'X<<Z', 'X>>Z']
    on_label = [("Z.upos",), ("f.label",), tuple()]
    for ns in nodes:
        for o in ordres:
            for extra in on_label:
                sadj[module_name((ns, o, extra, next(cpt)))] =\
                Sketch(Request('X[];Y[head]', ns, o), ["X.upos", "Y.upos"] + list(extra), 
                edge_between_X_and_Y, 
                no_edge_between_X_and_Y, "e.label")
    return apply_sketches(sadj, corpus, param)    

def append_head(g):
    for n in g:
        g[n]['head']='1'
    return g

def pack(s):
    if re.search("f[XYZ]", s):
        return 3
    if re.search("intermediate", s):
        return 2
    if "enhanced" in s:
        return 1
    return 0

def prepare_corpus(filename):
    corpus = Corpus(CorpusDraft(filename).apply(append_head))
    empty = Corpus(CorpusDraft(corpus).apply(clear_but_working))
    return corpus, empty

def basic_edges(g):
    def remove(e):
        return {'1' : e['1']}
    for n in g.sucs:
        g.sucs[n] = tuple((m,remove(e)) for m,e in g.sucs[n])
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='learner.py',
                                     description='Learn a grammar from sample files',
                                     epilog='Use connll files')

    parser.add_argument('train')
    parser.add_argument('-e', '--eval', default=None)
    parser.add_argument('--pickle', default=None)
    parser.add_argument('--rules', default=None)
    parser.add_argument('--details',default=False)
    args = parser.parse_args()
    if not args.eval:
        args.eval = args.train

    set_config("sud")
    param = {
        "base_threshold": 0.25,
        "valid_threshold": 0.90,
        "max_depth": 4,
        "min_samples_leaf": 5,
        "feat_value_size_limit": 10,
        "skip_features": ['xpos', 'upos', 'SpaceAfter', 'Shared', 'head'],
        "node_impurity": 0.2,
        "number_of_extra_leaves": 5, 
        "zipf_feature_criterion" : 0.95, 
        "min_occurrence_nb" : 10
    }
    corpus_gold, corpus_empty = prepare_corpus(args.train)
    if args.details:
        corpus_gold = Corpus(CorpusDraft(corpus_gold).apply(basic_edges))
    
    A = corpus_gold.count(Request('X<Y;e:X->Y'))
    A += corpus_gold.count(Request('Y<X;e:X->Y'))
    print("---target----")
    print(f"""number of edges within corpus: {corpus_gold.count(Request('e: X -> Y'))}""")
    print(f"number of adjacent relations: {A}")

    R0,L0 = adjacent_rules(corpus_gold, param)
    print(len(L0))
    R0e = refine_rules(R0, corpus_gold, param, 0)
    R0safe = R0e.safe_rules()
    for rn in R0safe:
        R0safe[rn].commands.append(Delete_feature("Y", "head"))

    print(f"number of rules len(R0e) = {len(R0safe)}")
    #turn R0e to a set of packages to speed up the process
    packages = {f'P{i}' : Package() for i in range(4)}
    for rn in R0safe:
        packages[f'P{pack(rn)}'][rn] = R0safe[rn]
    R00 = GRSDraft(packages)
    R00['main'] = "Seq(Onf(P0),Onf(P1),Onf(P2),Onf(P3))"
    if args.pickle:    
        pickle.dump( R00, open(args.pickle, "wb"))
    if args.rules:
        f = open(args.rules, "w")
        f.write( str(R00) )
        f.close()

    G00 = GRS(R00)
    web = grew_web.Grew_web()
    print(web.url())
    web.load_corpus(corpus_empty)

    currently_computed_corpus = get_best_solution(corpus_gold, corpus_empty, G00)
    print(currently_computed_corpus.edge_diff_up_to(corpus_gold))
    print(corpus_diff(currently_computed_corpus, corpus_gold))

