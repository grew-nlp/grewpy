from sklearn.tree import DecisionTreeClassifier
import numpy as np
import re
import itertools, random

# Use local grew lib
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")))

from grewpy.graph import Fs_edge
from grewpy import Request, GRSDraft, CorpusDraft, Corpus

def back_tree(T):
    """
    Given a classifier tree T, builds the parent relation: Y |-> (i,X) for 
    if Y is the i-th son of X in T
    returns also the set of leaves of T
    """
    back = dict()
    for node in range(T.node_count):
        g = T.children_left[node]
        d = T.children_right[node]
        if g >= 0: back[g] = (0, node)
        if d >= 0: back[d] = (1, node)
    return back

def branch(n, back):
    """
    starting from n, compute the branch from root node 0 to n
    by means of the back relation
    """
    branch = []
    while n != 0: #0 is the root node
        right, n = back[n] #right = is n the right son of its father?
        branch.append((n, right))
    branch.reverse() #get the branch in correct order
    return branch

def e_index(d):
    """
    given a collection d, maps an index to each element in d
    """
    cpt = iter(range(10000000))
    return {e : next(cpt) for e in d}

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

def build_request(clf, n, back, named_entities, request, idx2nkv, edge_idx, details):
    """
    build a request corresponding to a node n
    within the decision tree T
    back maps child node to its father
    request is the base request of the sketch
    if args.branch_details, we do not simplify the request
    """
    T = clf.tree_
    req = Request(request)  # builds a copy 
    root2n = branch(n, back) #
    criterions = [ (n, idx2nkv[T.feature[n]], r) for n,r in root2n]
    if details:
        for _,c,r in criterions:
            if r: req.with_(nkv2req(c))
            else: req.without(nkv2req(c))
        return req

    positives = [(n,c) for n,c,r in criterions if r]
    negatives = [(n,c) for n,c,r in criterions if not r]
    positive_features = {(nkv[0],nkv[1][0]) : nkv[1][1] for _,nkv in positives if isinstance(nkv,tuple)}
    positive_order = [(n,nkv) for n,nkv in positives if isinstance(nkv, str)]
    negative_features = [nkv for _,nkv in negatives if isinstance(nkv,tuple)]
    #now remove X[Gen=Masc] without {X[Gen=Fem]}
    good_negatives = [(m,(f,v)) for (m,(f,v)) in negative_features if (m,f) not in positive_features]
    negative_order = [(n,nkv) for n,nkv in negatives if isinstance(nkv, str)]

    if positive_order:
        for n,nkv in positive_order:
            if '<' in nkv:
                req.pattern(nkv)
            elif 'delta' in nkv:
                zz = nkv.split("_")
                N1,N2=zz[1],zz[2]
                tt = int(T.threshold[n]-0.5)
                req.pattern(f"{N1}[];{N2}[];delta({N1},{N2}) > {tt}")
            elif 'left' in nkv:
                z = nkv.split("_")[2]
                tt = int(T.threshold[n]-0.5)
                req.pattern(f"A[!upos];{z}[];delta(A,{z}) > {tt}")
            elif 'edge' in nkv:
                zz = nkv.split("_")
                f, e_idx = zz[1], int(zz[2])
                req.pattern(f'{f}.label="{edge_idx[e_idx]}"')
            else:
                raise NotImplementedError(f"{n, nkv}")
            
    for n in named_entities["nodes"]:
        nf = {k for (m,k) in positive_features if m==n}
        if nf:
            xf = ",".join( f'''{k}="{positive_features[(n,k)]}"''' for k in nf)
            req.append("pattern", f'{n}[{xf}]')
    for n,nkv in negative_order:
        if '<' in nkv:
            req.without( nkv)
        elif 'delta' in nkv:
            zz = nkv.split("_")
            N1,N2=zz[1],zz[2]
            tt = int(T.threshold[n]-0.5)
            req.pattern(f"{N1}[];{N2}[];delta({N1},{N2}) <= {tt}")
        elif 'left' in nkv:
            z = nkv.split("_")[2]
            tt = int(T.threshold[n]-0.5)
            req.pattern(f"A[!upos];{z}[];delta(A,{z}) <= {tt}")
        elif 'edge' in nkv:
                zz = nkv.split("_")
                f, e_idx = zz[1], int(zz[2])
                req.without(f'{f}.label="{edge_idx[e_idx]}"')
        else:
            raise NotImplementedError(f"{n, nkv}")

    for n,(k,v) in good_negatives:
        req.without(f'{n}[{k}="{v}"]')
    return req

def patterns(clf, idx2nkv, request, named_entities, edge_idx, param, y0, details):
    """
    list of patterns reaching prediction y0 within T
    """
    def loop(n): #list of nodes corresponding to outcome y0
        if np.argmax(T.value[n]) == y0 and T.impurity[n] < param['threshold']: return [n]
        if T.children_left[n] < 0: return [] #no children
        g = T.children_left[n]
        d = T.children_right[n]
        return loop(g) + loop(d)
    T = clf.tree_
    back = back_tree(T)
    res = [build_request(clf, n, back,  named_entities, request, idx2nkv, edge_idx, details) for n in loop(0)]
    return res

def order_constraints(nodes):
    """
    list all order constraints contained within nodes = X, Y, Z
    e.g. X<Y, Y<X,X<<Y
    """
    S1 = {f"{p}<{q}"  for p,q in itertools.permutations(nodes,2)}
    S2 = {f"{p}<<{q}" for p,q in itertools.permutations(nodes,2) if p < q}
    S3 = {f"delta_{p}_{q}" for p,q in itertools.permutations(nodes,2) if p < q}
    S4 = {f"left_pos_{p}" for p in nodes}
    S5 = {f"right_pos_{p}" for p in nodes}
    return S1 | S2 | S3 | S4 | S5

def nkv(corpus, skipped_features, named_entities, edge_idx, max_feature_values=50):
    """
    given a corpus, return the dictionnary  nkv -> index
    an nkv is either 
    a triple (n,k,v)
    - n in ('X', 'Y', ...)
    - k in ('Gen', 'upos', ...)
    - v in ('Fem', 'Det', ...)
    or nkv is an order constraint: nkv = "X << Y"
    or nkv is an edge label
    for each nk, we keep only max_feature_values 
    pattern_nodes = (X, Y, ...) list of nodes
    """
    observations = corpus.count_feature_values(exclude=list(skipped_features))
    feat_values_set = set()
    for k,vocc in observations.items():
        T = sorted([(occ,v) for v, occ in vocc.items()], reverse=True)
        feat_values_set |= {(k,v) for _,v in  T[:max_feature_values]}
    
    nkv = itertools.product(named_entities["nodes"], feat_values_set)
    nkv = set(nkv) | order_constraints(named_entities["nodes"])
    for f in named_entities["edges"]:
        nkv |= {f'edge_{f}_{idx}' for idx in edge_idx.values()}
    nkv_idx = e_index(nkv)
    order_idx = {k : v for k,v in nkv_idx.items() if isinstance(k,str)} #list of order constraints 
    return nkv_idx, order_idx

def edge_XY(graph, X2Name, edges, cpt, edge_idx, nkv_idx, X, y, order_idx):
    """
    for each match in the corpus, we fill X and y accordingly
    """
    for Xid, Xname in X2Name:
        for k,v in graph[Xid].items():
            if (Xname,(k,v)) in nkv_idx: X[(cpt,nkv_idx[(Xname,(k,v))])] += 1
    
    positions = {b : int(a) for a,b in X2Name}
    for n1,n2 in itertools.permutations(positions.keys(), 2):
        n12, n1_2, delta_n1_n2 = f'{n1}<{n2}',f'{n1}<<{n2}', f'delta_{n1}_{n2}'
        if n12 in order_idx:
            X[(cpt, order_idx[n12])] = 1 if positions[n1] - positions[n2] == -1 else 0
        if n1_2 in order_idx:
            X[(cpt, order_idx[n1_2])] = 1 if positions[n1] < positions[n2] else 0
        if delta_n1_n2 in order_idx:
            X[(cpt, order_idx[delta_n1_n2])] = positions[n2] - positions[n1]
    for a,b in X2Name:
        X[(cpt,order_idx[f'left_pos_{b}'])] = int(a)
        #X[(cpt,order_idx[f'right_pos_{b}'])] = len(graph) - int(a)
    for f in edges:
        X[(cpt,order_idx[f'edge_{f}_{edges[f]}'])] = 1

    Xn2id = {Xn : Xid for Xid,Xn in X2Name}
    e = graph.edge(Xn2id['X'], Xn2id['Y'])
    y[cpt] = edge_idx[e]
    return cpt+1

def build_Xy(gold, corpus, request, named_entities, edge_idx, nkv_idx, order_idx, ratio=0):
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
        nodes = [(match["matching"]["nodes"][n],n) for n in named_entities["nodes"]]
        edges = {
            f : edge_idx [ Fs_edge( match["matching"]["edges"][f]["label"] )]
            for f in named_entities["edges"]}
        cpt = edge_XY(graph, nodes, edges, cpt, edge_idx, nkv_idx, X, y, order_idx)
    if ratio == 0:
        selection = negatives
    else:
        weights = [ W(m) for m in negatives]
        selection = random.choices(negatives, weights=weights, k=N-cpt)
    for match in selection:
        graph = corpus[match['sent_id']]
        nodes = [(match["matching"]["nodes"][n],n) for n in named_entities["nodes"]]
        edges = {f :edge_idx [ Fs_edge(match["matching"]["edges"][f]["label"]) ] for f in named_entities["edges"]}
        cpt = edge_XY(graph, nodes, edges, cpt, edge_idx, nkv_idx, X, y, order_idx)
    return X,y

def observations(corpus_gold : Corpus, draft_gold : CorpusDraft, request : Request, edge_idx, named_entitites, param):
    """
    for each sample corresponding to the request, 
    return X[(sample,nkv)] -> 0/1 if nkv in sample
    and y[sample] = edge index
    """
    skipped_features = param['skipped_features']
    print("preprocessing")
    nkv_idx, order_idx = nkv(corpus_gold, skipped_features, named_entitites, edge_idx)
    X,y = build_Xy(corpus_gold,draft_gold, request, named_entitites, edge_idx, nkv_idx, order_idx, param['ratio'])
    return X,y,nkv_idx

def clf_dependency(dependency : int, X,y,idx2nkv,request : Request,named_entitites, idx_2_edge, param, depth, details):
    """
    build a decision tree classifier for the `dependency`
    with respect to observations `X`, `y`
    """
    clf = DecisionTreeClassifier(criterion="entropy", 
                                 min_samples_leaf=param["min_samples_leaf"], 
                                 max_depth=depth)
    print(f"learning {idx_2_edge[dependency]}")
    clf.fit(X, y == dependency)
    requests = patterns(clf, idx2nkv, request, named_entitites, idx_2_edge, param, 1, details) #1=good outcome
    return requests, clf
