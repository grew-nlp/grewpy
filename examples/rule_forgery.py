from sklearn.tree import DecisionTreeClassifier
import classifier
import numpy as np
import re
import itertools, random

# Use local grew lib
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")))
from grewpy.graph import Fs_edge
from grewpy.sketch import Sketch
from grewpy import Corpus, GRS
from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, CorpusDraft, Package, Delete_feature

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
    
def nkv2req(n):
    """
    transform an nkv to a request constraints
    """
    if isinstance(n,str): return n #it is an order constraints
    m,(feat, value) = n #it is a feature constraints
    if value == '"':
        value = '\\"'
    return f'{m}[{feat}="{value}"]'

def build_request(T, n, back, request_nodes, request, idx2nkv, details):
    """
    build a request corresponding to a node n
    within the decision tree T
    back maps child node to its father
    request is the base request of the sketch
    if args.branch_details, we do not simplify the request
    """
    req = Request(request)  # builds a copy 
    root2n = classifier.branch(n, back) #
    criterions = [ (idx2nkv[T.feature[n]], r) for n,r in root2n]
    if details:
        for n,r in criterions:
            if r: req.with_(nkv2req(n))
            else: req.without(nkv2req(n))
        return req

    positives = [n for n,r in criterions if r]
    negatives = [n for n,r in criterions if not r]
    positive_features = {(nkv[0],nkv[1][0]) : nkv[1][1] for nkv in positives if isinstance(nkv,tuple)}
    positive_order = [nkv for nkv in positives if isinstance(nkv, str)]
    negative_features = [nkv for nkv in negatives if isinstance(nkv,tuple)]
    #now remove X[Gen=Masc] without {X[Gen=Fem]}
    good_negatives = [(m,(f,v)) for (m,(f,v)) in negative_features if (m,f) not in positive_features]
    negative_order = [nkv for nkv in negatives if isinstance(nkv, str)]

    if positive_order:
        req.append("pattern", ";".join(positive_order))
    for n in request_nodes:
        nf = {k for (m,k) in positive_features if m==n}
        if nf:
            xf = ",".join( f'''{k}="{positive_features[(n,k)]}"''' for k in nf)
            req.append("pattern", f'{n}[{xf}]')
    for t in negative_order:
        req.without(t)
    for n,(k,v) in good_negatives:
        req.without(f'{n}[{k}="{v}"]')
    return req

def patterns(T, idx2nkv, request, nodes, param, y0, details):
    """
    list of patterns reaching prediction y0 within T
    """
    def loop(n): #list of nodes corresponding to outcome y0
        if np.argmax(T.value[n]) == y0 and T.impurity[n] < param['threshold']: return [n]
        if T.children_left[n] < 0: return [] #no children
        g = T.children_left[n]
        d = T.children_right[n]
        return loop(g) + loop(d)
    back = classifier.back_tree(T)
    return [build_request(T, n, back,  nodes, request, idx2nkv, details) for n in loop(0)]

def order_constraints(nodes):
    """
    list all order constraints contained within nodes = X, Y, Z
    e.g. X<Y, Y<X,X<<Y
    """
    S1 = {f"{p}<{q}"  for p,q in itertools.permutations(nodes,2)}
    S2 = {f"{p}<<{q}" for p,q in itertools.permutations(nodes,2) if p < q}
    return S1 | S2

def nkv(corpus, skipped_features, pattern_nodes, max_feature_values=50):
    """
    given a corpus, return the dictionnary  nkv -> index and indexes for order constraints
    an nkv is either 
    a triple (n,k,v)
    - n in ('X', 'Y', ...)
    - k in ('Gen', 'upos', ...)
    - v in ('Fem', 'Det', ...)
    or nkv is an order constraint: nkv = "X << Y"
    for each nk, we keep only max_feature_values 
    pattern_nodes = (X, Y, ...) list of nodes
    """
    observations = corpus.count_feature_values(exclude=list(skipped_features))
    feat_values_set = set()
    for k,vocc in observations.items():
        T = sorted([(occ,v) for v, occ in vocc.items()], reverse=True)
        feat_values_set |= {(k,v) for _,v in  T[:max_feature_values]}
    
    nkv = itertools.product(pattern_nodes, feat_values_set)
    nkv = set(nkv) | order_constraints(pattern_nodes)
    nkv_idx = classifier.e_index(nkv)
    order_idx = {k : v for k,v in nkv_idx.items() if isinstance(k,str)} #list of order constraints 
    return nkv_idx, order_idx

def edge_XY(graph, X2Name, cpt, edge_idx, nkv_idx, X, y, order_idx):
    for Xid, Xname in X2Name:
        for k,v in graph[Xid].items():
            if (Xname,(k,v)) in nkv_idx: X[(cpt,nkv_idx[(Xname,(k,v))])] += 1
    
    positions = {b : int(a) for a,b in X2Name}
    for n1,n2 in itertools.permutations(positions.keys(), 2):
        n12, n1_2 = f'{n1}<{n2}',f'{n1}<<{n2}'
        if n12 in order_idx:
            X[(cpt, order_idx[n12])] = 1 if positions[n1] - positions[n2] == -1 else 0
        if n1_2 in order_idx:
            X[(cpt, order_idx[n1_2])] = 1 if positions[n1] < positions[n2] else 0
    Xn2id = {Xn : Xid for Xid,Xn in X2Name}
    e = graph.edge(Xn2id['X'], Xn2id['Y'])
    y[cpt] = edge_idx[e]
    return cpt+1

def build_Xy(gold, corpus, request, free_nodes, edge_idx, nkv_idx, order_idx, ratio=0):
    """
    prepare data for learning
    corpus: from which patterns are extracted
    request: the sketch
    edge_idx: index for each edge label 
    nkv_idx : to each triple (node, feature, value) 
    ratio: 0 if we have to cover all samples in the corpus, otherwise, the ratio none/dependencies

    Return 
    X = matrix, for each pair of distinct nodes  (X,Y) in the corpus, 
    for each triple (n,k,v) in nkv_idx, X[nkv] = 1 if node n has feature (k,v)
    y = vector : y[X -> Y] = edge index between X and Y, None if there is no such edge
    """
    def W(match):
        nodes = match['matching']['nodes']
        delta = abs(int(nodes['X']) - int(nodes['Y']) )
        if delta == 0: return 0.0
        return 1/(delta+1)
    cpt = 0

    positives = gold.search(Request(request).with_('X->Y'))
    negatives = gold.search(Request(request).without('X->Y'))
    En = len(positives)+len(negatives)
    N = min(len(positives)*ratio, En) if ratio else En 
    X = np.zeros((N, len(nkv_idx)))
    y = np.zeros(N)
    cpt = 0
    for match in positives:
        graph = corpus[match['sent_id']]
        nodes = [(match["matching"]["nodes"][n],n) for n in free_nodes]
        cpt = edge_XY(graph, nodes, cpt, edge_idx, nkv_idx, X, y, order_idx)
    if ratio == 0:
        selection = negatives
    else:
        weights = [ W(m) for m in negatives]
        selection = random.choices(negatives, weights=weights, k=N-cpt)
    for match in selection:
        graph = corpus[match['sent_id']]
        nodes = [(match["matching"]["nodes"][n],n) for n in free_nodes]
        cpt = edge_XY(graph, nodes, cpt, edge_idx, nkv_idx, X, y, order_idx)
    return X,y

def observations(corpus_gold : Corpus,request : Request, nodes, param):
    """
    for each sample corresponding to the request, 
    return X[(sample,nkv)] -> 0/1 if nkv in sample
    and y[sample] = edge index
    """
    draft = CorpusDraft(corpus_gold)
    skipped_features = param['skipped_features']
    edges = {Fs_edge(x) for x in corpus_gold.count(Request("pattern{e:X->Y}"), ["e.label"]).keys()} | {None}
    #None for no dependency
    edge_idx = classifier.e_index(edges)
    print("preprocessing")
    nkv_idx, order_idx = nkv(corpus_gold, skipped_features, nodes)
    X,y = build_Xy(corpus_gold,draft, request, nodes, edge_idx, nkv_idx, order_idx, param['ratio'])
    return draft,X,y,edge_idx,nkv_idx

def clf_dependency(dependency : int, X,y,idx2nkv,request : Request,nodes,param, depth, details):
    """
    build a decision tree classifier for the `dependency`
    with respect to observations `X`, `y`
    """
    clf = DecisionTreeClassifier(criterion="entropy", 
                                 min_samples_leaf=param["min_samples_leaf"], 
                                 max_depth=depth)
    print("learning")
    clf.fit(X, y == dependency)
    requests = patterns(clf.tree_, idx2nkv, request, nodes, param, 1, details) #1=good outcome
    return requests, clf

"""
def anomaly(obsL,  threshold):
    '''
        L is a key within self
        return for L an edge and its occurrence evaluation 
        and number of total occurrences if beyond base_threshold
    '''
    s = sum(obsL.values())
    for x, v in obsL.items():
        if v > threshold * s and x:
            return (x, v, s)
    return None, None, None

def build_rules(sketch, observation, param, rule_name, verbose=False):
    '''
    search a rule adding an edge X -> Y, given a sketch  
    we build the clusters, then
    for each pair (X, upos=U1), (Y, upos=U2), we search for 
    some edge e occuring at least with probability base_threshold
    in which case, we define a rule R: 
    base_pattern /\ [X.upos=U1] /\ [Y.upos=U2] => add_edge X-[e]-Y
    '''
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
    '''
    Takes a request R, tries to find variants
    the result is the list of rules that refine pattern R
    for DEBUG, we return the decision tree classifier
    '''
    res = []
    matchings = corpus.search(R)
    '''
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
    '''
    return classifier.build_rules(matchings, corpus, R, param), None



def refine_rules(Rs, corpus, param, verbose=False):
    '''
    as above, but applies on a list of rules
    and filter only "correct" rules, see `param`
    return the list of refined version of rules Rs
    '''
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


'''
Learning sketches
'''
def edge_between_X_and_Y(P):
    return Request(P, 'e:X->Y')


def no_edge_between_X_and_Y(P):
    return Request(P).without('e:X-[^ANCESTOR]->Y')


def simple_sketch(r):
    return Sketch(r, ["X.upos", "Y.upos"], edge_between_X_and_Y, no_edge_between_X_and_Y, "e.label")


def apply_sketches(sketches, corpus, param):
    '''
    find rules from sketches
    '''
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
    '''
    build all adjacent rules. They are supposed to connect words at distance 1
    '''
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
    '''
    given a matchings corresponding to some request on the corpus,
    return a dict mapping (n,feature) =>(values)=>occurrences to its occurence number in matchings
    within the corpus. n : node name, feature like 'Gender', values like 'Fem'
    '''
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
"""