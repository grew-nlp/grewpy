from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import numpy as np
import argparse
import logging

from classifier import back_tree, e_index
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
    delete non working edges within g
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

def build_request(T, n, back, request, idx2nkv):
    req = Request(request)  # builds a copy 
    pos_order = list()
    neg_order = list()
    pos_feat = dict()
    neg_feat = dict()
    while n != 0: #0 is the root node
        right, n = back[n]
        Z = idx2nkv[T.feature[n]]
        if isinstance(Z, tuple): #it is a feature pattern
            m, feat, feat_value = Z
            feat_value = feat_value.replace('"', '\\"')
            Z = f'{m}[{feat}="{feat_value}"]'
            if right:
                pos_feat[(m,feat)] = Z
            else:
                if (m,feat) not in neg_feat:
                    neg_feat[(m,feat)] = []
                neg_feat[(m,feat)].append(Z)
        else:
            if right:
                pos_order.append(Z)
            else:
                neg_order.append(Z)
    if pos_order:
        req.append("pattern", ";".join(pos_order))
    if pos_feat:
        req.append("pattern", ";".join(pos_feat.values()))
    for n in neg_order:
        req.without(n)
    good_negatives = [(m,f) for (m,f) in neg_feat if (m,f) not in pos_feat]
    for n in good_negatives:
        for c in neg_feat[n]:
            req.without(c)
    return req

def forbidden_patterns(T, idx2e, idx2nkv, request, param):
    back, leaves = back_tree(T)
    internal_node = set()
    empty_patterns = []
    for n in leaves:
        if not idx2e[np.argmax(T.value[n])]: #no edge is majority
            while n and back[n][1] and T.impurity[ back[n][1] ] <= param['threshold']:
                #most general unifier
                n = back[n][1]
            if n and n not in internal_node and T.impurity[ n ] <= param['threshold']:
                internal_node.add(n)
                empty_patterns.append( build_request(T, n, back, request, idx2nkv))
    return empty_patterns

def nkv(corpus, skipped_features, max_feature_values=50):
    observations = dict()
    pair_number = 0
    for sid in corpus:
        graph = corpus[sid]
        pair_number += len(graph)**2 - len(graph)
        for n in graph:
            N = graph[n]  # feature structure of N
            for k, v in N.items():
                if k not in skipped_features:
                    if k not in observations:
                        observations[k] = dict()
                    observations[k][v] = observations[k].get(v, 0)+1
    ccc =  set()
    for k in observations:
        T = sorted([(n,v) for v,n in observations[k].items()], reverse=True)
        for n,v in T[:max_feature_values]:
            ccc.add((k,v))
    nkv = {('X',)+k for k in ccc} | {('Y',)+k for k in ccc} | {'X<Y', 'X>Y', 'X<<Y'}
    nkv_idx = e_index(nkv)
    return nkv_idx, pair_number

def build_Xy(corpus, skipped_features, edge_idx):
    print("preprocessing")    
    nkv_idx, XY_number = nkv(corpus, skipped_features)
    X = np.zeros((XY_number, len(nkv_idx)))
    y = np.zeros(XY_number)
    W = np.ones(XY_number)

    cpt = 0
    for sid in corpus:
        graph = corpus[sid]
        for Xn in graph:
            for Yn in graph:
                if Xn != Yn:
                    for k,v in graph[Xn].items():
                        if ('X',k,v) in nkv_idx:
                            X[(cpt,nkv_idx[('X',k,v)])] += 1
                    for k,v in graph[Yn].items():
                        if ('Y',k,v) in nkv_idx:
                            X[(cpt,nkv_idx[('Y',k,v)])] += 1
                    px = int(Xn)
                    py = int(Yn)
                    X[(cpt,nkv_idx['X<Y'])]  = 1 if ((px - py) == -1) else 0
                    X[(cpt,nkv_idx['X<<Y'])] = 1 if ((px - py) < 0) else 0
                    X[(cpt,nkv_idx['X>Y'])]  = 1 if ((px - py) == 1) else 0
                    e = graph.edge(Xn,Yn)
                    W[cpt] = 1 if e else 1
                    y[cpt] = edge_idx[e]
                    cpt += 1
    return X,y,W, nkv_idx
    
def zero_knowledge_voids(corpus_gold, args, param):
    draft = CorpusDraft(corpus_gold)
    skipped_features = param['skipped_features']
    edges = {Fs_edge(x) : 1 for x in corpus_gold.count(Request("e:X->Y"), ["e.label"]).keys()} | {None : 1.1}
    edge_idx = e_index(edges)

    X,y,W,nkv_idx = build_Xy(draft, skipped_features, edge_idx)
    clf = DecisionTreeClassifier(criterion="gini", 
                                 min_samples_leaf=param["min_samples_leaf"], 
                                 max_depth=8,
                                 class_weight={ edge_idx[e] : edges[e] for e in edges})
    print("learning")
    clf.fit(X, y, sample_weight=W)
    idx2nkv = {v:k for k,v in nkv_idx.items()}
    idx2e = {v:k for k,v in edge_idx.items()}
    voids = forbidden_patterns(clf.tree_, idx2e, idx2nkv, Request("X[];Y[];e:X->Y"), param)
    if args.forbidden:
        f = open(args.forbidden, "w")
        for pattern in voids:
            f.write(f"###\n{str(pattern)}\n")
        f.close()
    print("testing patterns")
    found = False
    for request in voids:
        lines = corpus_gold.search(request)
        if lines:
            print(f"forbidden pattern\n{request}\n")
            print(f"""sent_id : {",".join(f"{match['sent_id']}" for match in lines)}
""")
            found = True
    if not found:
        print("self verification: no forbidden patterns found")
    if args.web:
        empty_rules = [ Rule( request, Commands()) for request in voids]
        empty_rules = { f"rule{i}" : empty_rules[i] for i in range(len(empty_rules))} #give a name
        web = grew_web.Grew_web()
        print(f"go to {web.url()}")
        web.load_corpus(corpus_gold)
        web.load_grs(GRS(empty_rules))

def parse_request(filename):
    """
    parsing a (anti-)pattern file 
    """
    excluded = []
    current_pattern = None
    with open (filename) as f: #we parse the file
        for line in f:
            if '###' in line:
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
    excluded = parse_request(args.forbidden)
    print("testing patterns")
    found = False
    for req in excluded:
        lines = corpus.search(req)
        if lines:
            found = True
            print(f"found forbidden pattern\n{req}")
            print(f"""sent_id : {",".join(f"{match['sent_id']}" for match in lines)}
""")
    if not found:
        print("nothing found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='antipattern.py',
                                     description='Learn/Test antipattern on a corpus')
    parser.add_argument('action', help="action in [learn, verify]")
    parser.add_argument('corpus', help='a conll file')
    parser.add_argument('-f', '--forbidden', default=None, help="filename for the antipatterns")
    parser.add_argument('-t', '--threshold', default=1e-4, help="minimal threshold to consider a node as pure")
    parser.add_argument('-s','--nodetails',action="store_true", help="simplifies edges: replace comp:obl by comp")
    parser.add_argument('-w', '--web', action="store_true", help="for a debugging session")
    parser.add_argument('-d', '--depth', default=8, help="depth of the binary decision tree")
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
