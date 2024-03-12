from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
import numpy as np
import argparse
import logging
import itertools

from classifier import back_tree, e_index, branch
# Use local grew lib
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from grewpy import Request, Rule, Commands, CorpusDraft
from grewpy import Corpus, GRS, set_config
from grewpy import grew_web
from grewpy.graph import Fs_edge

def body(line): #get the body of a request"""
    return line[line.index('{')+1:].strip().strip('}')

def clear_but_working(g):
    """
    delete edges within g
    """
    g.sucs = {n : [] for n in g.sucs}
    return (g)

def basic_edges(g):
    """
    change edges {1:comp, 2:obl} to {1:comp}
    """
    def remove(e):
        return Fs_edge({'1': e['1']})
    for n in g.sucs:
        g.sucs[n] = tuple((m, remove(e)) for m, e in g.sucs[n])
    return g

def append_nkv(n, req, with_or_without):
    """
    append an nkv to req, with_or_with = "pattern" or "without"
    """
    if isinstance(n, tuple):
        m,(feat, value) = n
        req.append(with_or_without, f'{m}[{feat}="{value}"]')
    else:
        req.append(with_or_without,n)

def build_request(T, n, back, request, idx2nkv, args):
    """
    build a request correpsonding to a node n
    within the decision tree T
    back maps child node to its father
    request is the base request of the sketch
    if args.branch_details, we do not simplify the request
    """
    req = Request(request)  # builds a copy 
    root2n = branch(n, back) #
    criterions = [ (idx2nkv[T.feature[n]], r) for n,r in root2n]
    if args.branch_details:
        for n,r in criterions:
            req_c = "pattern" if r else "without"
            append_nkv(n, req, req_c)
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
    nodes = ('X','Y')
    for n in nodes:
        nf = {k for (m,k) in positive_features if m==n}
        if nf:
            xf = ",".join( f'''{k}="{positive_features[(n,k)]}"''' for k in nf)
            req.append("pattern", f'{n}[{xf}]')
    for t in negative_order:
        req.without(t)
    for n,(k,v) in good_negatives:
        req.without(f'{n}[{k}="{v}"]')
    return req

def patterns_leaves(T, idx2nkv, request, param, y0, args):
    """
    list patterns reaching prediction y0 within T
    """
    back, leaves = back_tree(T)
    internal_node = set()
    empty_patterns = []
    for n in leaves:
        if np.argmax(T.value[n]) == y0: #y0 is majority
            while n and back[n][1] and T.impurity[ back[n][1] ] <= param['threshold']:
                #most general unifier
                n = back[n][1]
            if n and n not in internal_node and T.impurity[ n ] <= param['threshold']:
                internal_node.add(n)
                empty_patterns.append( build_request(T, n, back, request, idx2nkv, args))
    return empty_patterns

def nkv(corpus, skipped_features, pattern_nodes, max_feature_values=50):
    """
    given a corpus, return the dictionnary  nkv_idx -> index 
    an nkv to an index
    an nkv is either 
    a triple (n,k,v)
    - n in ('X', 'Y', ...)
    - k in ('Gen', 'upos', ...)
    - v in ('Fem', 'Det', ...)
    or nkv is an order constraint: nkv = "X << Y"
    for each nk, we keep only max_feature_values 
    pattern_nodes = (X, Y, ...) list of nodes
    pair_number = corpus.count( "X << Y")
    """
    observations = dict() #maps an nk => v => nb of occurrences
    pair_number = 0
    for graph in corpus.values():
        pair_number += len(graph)**2 - len(graph)
        for N in graph.features.values():
            for k, v in N.items():
                if k not in skipped_features:
                    if k not in observations:
                        observations[k] = dict()
                    observations[k][v] = observations[k].get(v, 0)+1
    ccc =  set()
    for k in observations:
        T = sorted([(occurrences,v) for v, occurrences in observations[k].items()], reverse=True)
        for _,v in T[:max_feature_values]:
            ccc.add((k,v))
    nkv = itertools.product(pattern_nodes, ccc)
    nkv = set(nkv) |  {'X<Y', 'X>Y', 'X<<Y'}
    nkv_idx = e_index(nkv)
    return nkv_idx, pair_number

def edge_XY(graph, Xn,Yn, cpt, edge_idx, nkv_idx, X, y, xpy, xby, xgy):
    for k,v in graph[Xn].items():
        if ('X',(k,v)) in nkv_idx: X[(cpt,nkv_idx[('X',(k,v))])] += 1    
    for k,v in graph[Yn].items():
        if ('Y',(k,v)) in nkv_idx: X[(cpt,nkv_idx[('Y',(k,v))])] += 1
    px, py = int(Xn), int(Yn)
    X[(cpt,xpy)]  = 1 if ((px - py) == -1) else 0
    X[(cpt,xby)] = 1 if ((px - py) < 0) else 0
    X[(cpt,xgy)]  = 1 if ((px - py) == 1) else 0
    e = graph.edge(Xn,Yn)
    y[cpt] = edge_idx[e]
    return cpt+1

def build_Xy(gold, corpus, edge_idx, nkv_idx, sample_size, full=True):
    """
    prepare data for the decision tree
    corpus: from which patterns are extracted
    edge_idx: index for each edge label 
    nkv_idx : to each triple (node, feature, value) 
    sample_size: size of the output
    full: True if we have to cover all samples in the corpus, otherwise, the ratio 

    Return 
    X = matrix, for each pair of distinct nodes  (X,Y) in the corpus, 
    for each triple (n,k,v) in nkv_idx, X[nkv] = 1 if node n has feature (k,v)
    y = vector : y[X -> Y] = edge index between X and Y, None if there is no such edge
    """
    xpy, xgy, xby = nkv_idx['X<Y'], nkv_idx['X>Y'], nkv_idx['X<<Y']
    cpt = 0
    if full is True:    
        X = np.zeros((sample_size, len(nkv_idx)))
        y = np.zeros(sample_size)
        for graph in corpus.values():
            for Xn,Yn in itertools.permutations(graph, 2):
                cpt = edge_XY(graph, Xn,Yn, cpt, edge_idx, nkv_idx, X, y, xpy, xby, xgy)
        return X,y
    dependencies = gold.search(Request("X[];Y[];e:X->Y"), ["e.label"])
    samples_nb = sum([len(v) for v in dependencies])
    X = np.zeros((int(samples_nb/full), len(nkv_idx)))
    y = np.zeros(int(samples_nb/full))    
    for dep, matchings in dependencies.items():
        for match in  matchings:
            Xn, Yn = [int(match['matching']['nodes'][N]) for N in ('X','Y')]
            cpt = edge_XY(graph, Xn,Yn, cpt, edge_idx, nkv_idx, X, y, xpy, xby, xgy)

def check(corpus, excluded, edge):
    """
    return true if no pattern within excluded has an edge 
    """
    found = False
    for request in excluded:
        if edge: request = request.without(f'X-["{edge}"]->Y') 
        else: request.append("pattern", "e:X->Y")
        lines = corpus.search(request)
        if lines:
            found = True
            print(f"forbidden pattern\n{request}\n")
            print(f"""sent_id : {",".join(f"{match['sent_id']}" for match in lines)}\n""")
    return not found

def fit_dependency(clf,X,y,edge_idx,args):
    print("learning")
    if args.dependency:
        edg = Fs_edge(args.dependency)
        if edg not in edge_idx:
            print(f"Issue: {args.dependency} is not a dependency of the corpus (e.g. subj)")
            sys.exit(2)
        else:
            y = (y == edge_idx[edg])
    else:
        y = (y == edge_idx[None])
    clf.fit(X, y)

   
def zero_knowledge_voids(corpus_gold, args, param):
    draft = CorpusDraft(corpus_gold)
    skipped_features = param['skipped_features']
    edges = {Fs_edge(x) for x in corpus_gold.count(Request("e:X->Y"), ["e.label"]).keys()} | {None}
    edge_idx = e_index(edges)
    print("preprocessing")    
    nkv_idx, XY_number = nkv(draft, skipped_features, ('X','Y'))
    idx2nkv = {v:k for k,v in nkv_idx.items()}
    X,y = build_Xy(corpus_gold,draft, edge_idx, nkv_idx, XY_number, True)
    clf = DecisionTreeClassifier(criterion="entropy", 
                                 min_samples_leaf=param["min_samples_leaf"], 
                                 max_depth=args.depth)
    fit_dependency(clf, X, y, edge_idx, args)
    if args.export_tree:
        export_graphviz(clf, out_file=args.export_tree, 
                        feature_names=[str(idx2nkv[i]) for i in range(len(idx2nkv))])
    
    requests = patterns_leaves(clf.tree_, idx2nkv, Request("X[];Y[]"), param, 1, args)
    if args.file:
        with open(args.file, "w") as f:
            for pattern in requests:
                f.write(f"%%%\n{str(pattern)}\n")
    print("testing patterns")            
    if check(corpus_gold, requests, args.dependency):
        print("self verification: no forbidden patterns found")
    if args.web:
        empty_rules = [ Rule( request, Commands()) for request in requests]
        empty_rules = { f"rule{i}" : empty_rules[i] for i in range(len(empty_rules))} #give a name
        web = grew_web.Grew_web()
        print(f"go to {web.url()}")
        web.load_corpus(corpus_gold)
        web.load_grs(GRS(empty_rules))


def parse_request(filename):
    """
    parsing an (anti-)pattern file 
    """
    excluded = []
    current_pattern = None
    with open (filename) as f: #we parse the file
        for line in f:
            if '%%%' in line:
                if current_pattern:
                    excluded.append(current_pattern)
                current_pattern = Request()
            else:
                if "pattern" in line:
                    current_pattern.append("pattern", body(line))
                elif "without" in line:
                    current_pattern.without(body(line))
                else:
                    logging.critical(f"oups {line} is not in a pattern")
    if current_pattern:
        excluded.append(current_pattern)
    return excluded


def verify(corpus, args):
    """
    for requests within args.file, verifies whether we found them
    in the corpus
    """
    excluded = parse_request(args.file)
    print("testing patterns")
    if check(corpus,excluded):
        print("nothing found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='antipattern.py',
                                     description='Learn/Test antipattern on a corpus')
    parser.add_argument('action', help="action in [learn, verify]")
    parser.add_argument('corpus', help='a conll file')
    parser.add_argument('-f', '--file', default=None, help="filename for the patterns")
    parser.add_argument('-t', '--threshold', default=1e-10, type=float, help="minimal threshold to consider a node as pure")
    parser.add_argument('-s','--nodetails',action="store_true", help="simplifies edges: replace comp:obl by comp")
    parser.add_argument('-w', '--web', action="store_true", help="for a debugging session")
    parser.add_argument('-d', '--depth', default=8, help="depth of the binary decision tree")
    parser.add_argument('-b', '--branch_details', action="store_true", help="if set, requests are fully decomposed")   
    parser.add_argument('--export_tree', default='', help='export classifier tree as a dot file')
    parser.add_argument('--dependency', default='', help='export classifier tree as a dot file')
    args = parser.parse_args()
    set_config("sud")
    param = {
        "min_samples_leaf": 5,
        "skipped_features":  {'xpos', 'SpaceAfter', 'Shared', 'textform', 'Typo', 'form', 'wordform', 'CorrectForm'},
        "node_impurity": 0.15,
        "threshold" : args.threshold,
        "tree_depth" : args.depth
    }
    corpus_gold = Corpus(args.corpus)
    if args.nodetails:
        corpus_gold = Corpus(CorpusDraft(corpus_gold).apply(basic_edges))
    if args.action == 'learn':
        zero_knowledge_voids(corpus_gold, args, param)
    elif args.action == 'verify':
        verify(corpus_gold, args)
    else:
        logging.critical('action {args.action} is not known. Use either learn or verify.')
