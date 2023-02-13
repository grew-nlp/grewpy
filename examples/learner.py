from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np
import re
import argparse

# Use local grew lib
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, CorpusDraft
from grewpy import Corpus, GRS, set_config
from grewpy.observation import Observation
from grewpy.sketch import Sketch

import classifier

is_working = lambda e : "rank" not in e

def remove_rank(e):#remove rank within edges
    if "rank" in e:
        return tuple(sorted(list((k, v) for k, v in e.items() if k != "rank")))

def edge_equal_up_to_rank(e1, e2):
    return remove_rank(e1) == remove_rank(e2)

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
            clauses = []
            for it in val.split(","):
                a, b = it.split("=")
                clauses.append(f'{edge_name}.{a}="{b}"')
            return ";".join(clauses)
        return f"{crit}={val}"
    rules = WorkingGRS()
    for parameter in observation:
        x, v, s = observation.anomaly(parameter, param["base_threshold"])
        if x:
            extra_pattern = [crit_to_request(crit, val) for (
                crit, val) in zip(sketch.cluster_criterion, parameter)]
            P = Request(sketch.P, *extra_pattern)
            x = x[0].replace(f"rank=_", f'rank="{rank_level}"')
            c = Add_edge("X", x, "Y")
            R = Rule(P, Commands(c))
            rn = re.sub("[.,=\"]", "",
                            f"_{'_'.join(parameter)}_{rule_name}")
            rules[rn] = (R, (x, (v, s)))
    return rules


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
                e["rank"] = rank
                rule = Rule(request, Commands(Add_edge("X", e, "Y")))
                res.append(rule)
    return res, clf


def refine_rules(Rs, corpus, param, rank, debug=False):
    """
    as above, but applies on a list of rules
    and filter only "correct" rules, see `param`
    return the list of refined version of rules Rs
    """
    Rse = GRSDraft()
    for rule_name in Rs.rules():
        R = Rs[rule_name]
        v, s = Rs.eval[rule_name][1]
        if v/s < param["valid_threshold"] or v < param["min_occurrence_nb"]:
            new_r, clf = refine_rule(R.request, corpus, param, rank)
            if len(new_r) >= 1:
                cpt = 1
                if debug:
                    print("--------------------------replace")
                    print(R)
                for r in new_r:
                    if debug:
                        print("by : ")
                        print(r)
                    Rse[f"{rule_name}_enhanced{cpt}"] = r
                    cpt += 1
        else:
            Rse[rule_name] = R
    return Rse


"""
Operations on the corpus, compute SPAN, ANCESTOR, remove working edges, label edges with default rank
"""
def add_span(g):
    left, right = {i: i for i in g}, {i: i for i in g}
    todo = [i for i in g]
    g._sucs = {i : g._sucs.get(i,[]) for i in g}
    while todo:
        n = todo.pop(0)
        for s, _ in g.sucs[n]:
            if g.lower(left[s], left[n]):
                left[n] = left[s]
                todo.append(n)
            if g.greater(right[s], right[n]):
                right[n] = right[s]
                todo.append(n)
    for i in g:
        g.sucs[i] += [(left[i], {'1': 'LEFT_SPAN'}),
                      (right[i], {'1': 'RIGHT_SPAN'})]
    return g


def add_ancestor_relation(g):
    """
    add the ancestor relation to g (supposed to be a tree)
    """
    todo = [(i, i) for i in g]
    while todo:
        n, m = todo.pop()
        for s, e in g.sucs[m]:
            if not is_working(e):
                g.sucs[n].append((s, {"1": "ANCESTOR"}))
                todo.append((n, s))
    return g


def clear_but_working(g):
    """
    delete non working edges within g
    """
    g.sucs = {n : [(s,e) for s,e in g.sucs[n] if is_working(e)] for n in g.sucs}
    return (g)

def add_rank(g):
    """
    add rank=_ to any edge 
    """
    def add_null_rank(e):
        if isinstance(e, str):
            return {"1": e, "rank": "_"}
        e["rank"] = "_"
        return e
    for n in g:
        if n in g.sucs:
            g.sucs[n] = [(m, add_null_rank(e)) for (m, e) in g.sucs[n]]
    return g

"""
Operations on computations
"""

def get_best_solution(corpus_gold, corpus_start, grs):
    """
    grs is a GRSDraft
    return the best solution using the grs with respect to the gold corpus
    the grs does not need to be confluent. We take the best solution with 
    respect to the f-score
    """
    def f_score(t):
        return t[0]/math.sqrt((t[0]+t[1])*(t[0]*t[2])+1e-20)

    corpus_draft = CorpusDraft(corpus_start)
    for sid in corpus_gold:
        gs = grs.run(corpus_start[sid], 'main')
        best_fscore = 0
        for g in gs:
            fs = f_score(g.edge_diff_up_to(corpus_gold[sid], remove_rank))
            if fs > best_fscore:
                best_fscore = fs
                corpus_draft[sid] = g
    return corpus_draft

def update_gold_rank(corpus_gold, computed_corpus, rank):
    """
    build a copy of corpus_gold but with a rank update
    """
    new_gold = CorpusDraft(corpus_gold)
    for sid in new_gold:
        g_gold = new_gold[sid]
        g_new = computed_corpus[sid]
        for n in g_gold.sucs:
            new_sucs = []
            for m,e in g_gold.sucs[n]:
                if is_working(e):
                    new_sucs.append((m,e))
                else:
                    eprime = g_new.edge_up_to(
                        n, m, lambda eprime: edge_equal_up_to_rank(e, eprime))
                    if eprime:
                        new_sucs.append((m, eprime))
                    else:
                        new_sucs.append((m,e))
            g_gold.sucs[n] = new_sucs
    return new_gold

def remove_wrong_edges(corpus, corpus_gold):
    """
    remove from corpus those edges not in corpus_gold
    edges are compared up to rank
    """
    def extract_edge(edges, e):
        """
        return an edge in edges that is equal to e up to rank
        """
        for e1 in edges:
            if edge_equal_up_to_rank(e1, e):
                return e
        return None
    new_corpus = CorpusDraft(corpus)
    for sid in new_corpus:
        g_gold = corpus_gold[sid]
        g_corp = new_corpus[sid]
        for n in g_corp:
            old_edges = g_corp.sucs[n]
            g_corp.sucs[n] = []
            for (m, e) in old_edges:
                gold_edges = g_gold.edges(n, m)
                if extract_edge(gold_edges, e):
                    g_corp.sucs[n].append((m, e))
    return new_corpus

"""
Learning sketches
"""

def edge_between_X_and_Y(P):
    return Request(P, 'e:X->Y;e.rank="_"')

def no_edge_between_X_and_Y(P):
    return Request(P).without('e:X-[^LEFT_SPAN|RIGHT_SPAN|ANCESTOR]->Y')

def simple_sketch(r):
    return Sketch(r, ["X.upos", "Y.upos"], edge_between_X_and_Y, no_edge_between_X_and_Y, "e.label")

def apply_sketches(sketches, corpus, param, rank_level):
    """
    find rules from sketches
    """
    rules = WorkingGRS()
    for sketch_name in sketches:
        sketch = sketches[sketch_name]
        rules |= build_rules(sketch, sketch.cluster(corpus), param, sketch_name, rank_level=rank_level)
    return rules

def adjacent_rules(corpus: Corpus, param) -> WorkingGRS:
    """
    build all adjacent rules. They are supposed to connect words at distance 1
    """
    sadj = dict()    
    sadj["adjacent_lr"] = simple_sketch(Request("X[];Y[];X<Y"))
    sadj["adjacent_rl"] = simple_sketch(Request("X[];Y[];Y<X"))
    sadj["no_intermediate_1"] = simple_sketch(Request("X[];Y[];X<<Y").without("Z[];X<<Z;Z<<Y;X.upos=Z.upos"))
    sadj["no_intermediate_2"] = simple_sketch(Request("X[];Y[];X<<Y").without("Z[];X<<Z;Z<<Y;Y.upos=Z.upos"))
    sadj["no_intermediate_3"] = simple_sketch(Request("X[];Y[];Y<<X").without("Z[];Y<<Z;Z<<X;Y.upos=Z.upos")) 
    sadj["no_intermediate_4"]=simple_sketch(Request("X[];Y[];Y<<X").without("Z[];Y<<Z;Z<<X;X.upos=Z.upos"))
    return apply_sketches(sadj, corpus, param, 0)    

def span_sketch(r):
    return Sketch(r, ["X.upos", "Y.upos","Z.upos"], edge_between_X_and_Y, no_edge_between_X_and_Y, "e.label")

def span_rules(corpus, param):
    sketches = dict()
    sketches["span_Zlr"] = span_sketch(Request("X[];Y[];X -[LEFT_SPAN]->Z;Z<Y;"))
    sketches["span_Zrl"] = span_sketch(Request("X[];Y[];X -[RIGHT_SPAN]->Z;Y<Z"))
    sketches["span_T1"] = simple_sketch(Request("X[];Y[];Y -[LEFT_SPAN]->T;X<T"))
    sketches["span_T2"] = simple_sketch(Request("X[];Y[];Y -[RIGHT_SPAN]->T;T<X"))
    sketches["span_ZTlr"] = simple_sketch(Request("X[];Y[];Y -[LEFT_SPAN]->T;X-[LEFT_SPAN]->Z;Z<T"))
    sketches["span_ZTrl"] = simple_sketch(Request("X[];Y[];Y -[RIGHT_SPAN]->T;X-[RIGHT_SPAN]->Z;T<Z"))
    return apply_sketches(sketches, corpus, param, 0)


def ancestor_rules(corpus, param):
    sketches = dict()
    sketches["ancestor_zy"] = span_sketch(Request("X[];Y[];X-[ANCESTOR]->Z;Z<Y"))
    sketches["ancestor_yz"] = span_sketch(Request("X[];Y[];X-[ANCESTOR]->Z;Y<Z"))
    sketches["ancestor_xz"] = span_sketch(Request("X[];Y[];Y-[ANCESTOR]->Z;X<Z"))
    sketches["zy_ancestor"] = span_sketch(Request("X[];Y[];Z-[ANCESTOR]->X;Z<Y"))
    sketches["yz_ancestor"] = span_sketch(Request("X[];Y[];Z-[ANCESTOR]->X;Y<Z"))
    sketches["zx_ancestor"] = span_sketch(Request("X[];Y[];Z-[ANCESTOR]->Y;Z<X"))
    sketches["xz_ancestor"] = span_sketch(Request("X[];Y[];Z-[ANCESTOR]->Y;X<Z"))
    sketches["span_ancestor_zy"] = span_sketch(Request("X[];Y[];Y-[LEFT_SPAN]->T;X-[ANCESTOR]->Z;Z<T"))
    sketches["span_ancestor_yz"]= span_sketch(Request("X[];Y[];Y-[RIGHT_SPAN]->T;X-[ANCESTOR]->Z;T<Z"))
    return apply_sketches(sketches, corpus, param, 0)

def rank_n_plus_one(corpus_gold, param, rank_n):
    """
    build rules for corpus
    """
    rules = WorkingGRS()
    corpus = Corpus(corpus_gold)
    nodes = ['f:X -> Z', 'f:Y -> Z', 'f:Z->X', 'f:Z->Y']
    ordres = ['X<Y', 'X>Y', 'Z<Y', 'Z>Y', 'X<Z', 'X>Z']
    sketches = dict()
    for ns in nodes:
        for o in ordres:
            for rank in range(0,rank_n+1):
                sketches[(ns,o,rank)] = Sketch(Request('X[];Y[]', ns, o, f'f.rank="{rank}"'), 
                ["X.upos", "Y.upos"], edge_between_X_and_Y, no_edge_between_X_and_Y, "e.label")
    
    cpt = 1
    for ns in nodes:
        for o in ordres:
            obs = Observation()
            for rank in range(0, rank_n+1):
                obs |= sketches[(ns,o,rank)].cluster(corpus)
            rules |= build_rules(sketches[(ns,o,rank_n)], obs, param, f"rank_{rank_n}_{cpt}", 
            rank_level=rank_n+1)
            cpt += 1
    return rules

def rule_analysis(Rs, DRs, corpus):
    """
    find rule agreement/disagreement
    """
    applications = CorpusDraft()
    for R in DRs.rules():
        applications[R] = CorpusDraft({sid : Rs.run(corpus[sid], f'Onf({R})')[0] for sid in corpus})
    matrix = dict()
    for R in DRs.rules():
        for S in DRs.rules():
            matrix[(R,S)] = applications[R].edge_diff_up_to(applications[S], remove_rank)
    return matrix


def prepare_corpus(filename):
    corpus = CorpusDraft(filename)
    corpus = corpus.apply(add_rank)  # append rank label on edges
    corpus = corpus.apply(add_span)  # span
    corpus = Corpus(corpus.apply(add_ancestor_relation))
    empty = Corpus(CorpusDraft(corpus).apply(clear_but_working))
    return corpus, empty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='learner.py',
                                     description='Learn a grammar from sample files',
                                     epilog='Use connll files')

    parser.add_argument('train')
    parser.add_argument('-e', '--eval', default=None)
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
        "skip_features": ['xpos', 'upos', 'SpaceAfter'],
        "node_impurity": 0.2,
        "number_of_extra_leaves": 5, 
        "zipf_feature_criterion" : 0.95, 
        "min_occurrence_nb" : 10
    }
    corpus_gold, corpus_empty = prepare_corpus(args.train)

    packages = []#list the packages according to their rank 
    draft_packages = [] #list the draft versions of the packages

    A = corpus_gold.count(Request('X<Y;e:X->Y;e.rank="_"'))
    A += corpus_gold.count(Request('Y<X;e:X->Y;e.rank="_"'))
    print("---target----")
    print(f"""number of edges within corpus: {corpus_gold.count(Request('e: X -> Y;e.rank="_"'))}""")
    print(f"number of adjacent relations: {A}")

    R0 = adjacent_rules(corpus_gold, param)
    R0e = refine_rules(R0, corpus_gold, param, 0)
    print(f"adjacent rules len(R0e) = {len(R0e)}")
    R0e_test = GRS(R0e.safe_rules().onf())
    c = get_best_solution(corpus_gold, corpus_empty, R0e_test)
    print(c.edge_diff_up_to(corpus_gold,remove_rank))

    Rs0 = span_rules(corpus_gold, param)
    Rs0e = refine_rules(Rs0, corpus_gold, param, 0)
    Rs0e_t = GRS(Rs0e.safe_rules().onf())
    print(f"span rules: len(Rs0) = {len(Rs0e)}")
    c = get_best_solution(corpus_gold, corpus_empty, Rs0e_t)
    print(c.edge_diff_up_to(corpus_gold,remove_rank))

    Ra0 = ancestor_rules(corpus_gold, param)
    Rae0 = refine_rules(Ra0, corpus_gold, param, 0)
    Ra0e_t = GRS(Rae0.safe_rules().onf())
    print(f"ancestor rules after refinement {len(Rae0)}")
    c = get_best_solution(corpus_gold, corpus_empty, Ra0e_t)
    print(c.edge_diff_up_to(corpus_gold,remove_rank))

    draft_packages.append(GRSDraft(R0e | Rs0e | Rae0).safe_rules().onf())
    packages.append(GRS(draft_packages[-1]))

    currently_computed_corpus = get_best_solution(corpus_gold, corpus_empty, packages[0])
    print(currently_computed_corpus.edge_diff_up_to(corpus_gold, remove_rank))

    """
    confusion_matrix = rule_analysis(packages[0], draft_packages[0], corpus_empty)
    f = open("matrix.csv", "w")
    f.write(";" + ";".join(draft_packages[0].rules()))
    for R in draft_packages[0].rules():
        f.write(f"\n{R};")
        f.flush()
        for S in draft_packages[0].rules():
            a = confusion_matrix[(R, S)]
            f.write(f"{a['common']}/{a['left']}/{a['right']}")
            f.write(";")
            f.flush()
    f.close()
    """


    for rank in range(1, 4):
        corpus_gold_after_step = update_gold_rank(corpus_gold, currently_computed_corpus, rank)
        Rnext = rank_n_plus_one(corpus_gold_after_step, param, rank - 1)
        Rnexte = refine_rules(Rnext, corpus_gold, param, rank)
        draft_packages.append(Rnexte.safe_rules().onf())
        packages.append(GRS(draft_packages[-1]))
        currently_computed_corpus = get_best_solution(corpus_gold, currently_computed_corpus, packages[rank])
        print(f"-----------Rank {rank} rules : {len(Rnexte)}")
        print((currently_computed_corpus.edge_diff_up_to(corpus_gold,remove_rank)))

    print("------Now testing on the evaluation corpus----------")
    corpus_gold_eval, computed_corpus_eval = prepare_corpus(args.eval)
    for rank in range(4):
        print(f"--------at rank {rank} ------------")
        computed_corpus_eval = get_best_solution(corpus_gold_eval, computed_corpus_eval, packages[rank])
        print(computed_corpus_eval.edge_diff_up_to(corpus_gold_eval, remove_rank))
        computed_corpus_eval = remove_wrong_edges(computed_corpus_eval, corpus_gold_eval)


    for rank in range(4):
        print(f"--------R{rank} rules------")
        print(f"{draft_packages[rank]}")
