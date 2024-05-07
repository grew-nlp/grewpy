import math
import numpy as np
import re
import argparse
import pickle
from dataclasses import dataclass

from rule_forgery import WorkingGRS, observations, clf_dependency
# Use local grew lib
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from grewpy import Request, Rule, Commands, Command, Add_edge, GRSDraft, CorpusDraft, Package, Delete_feature
from grewpy import Corpus, GRS, set_config
from grewpy import grew_web
from grewpy.graph import Fs_edge

def add_rule_name(wgrs):
    for rn in wgrs:
        wgrs[rn].commands.append(Command(f'Y.rule="{rn}"'))

def clear_but_working(g):
    """
    delete non working edges within g
    """
    g.sucs = {n : [] for n in g.sucs}
    return (g)

def append_head(g):
    """
    each node in g get a head feature = 1
    """
    for n in g: g[n]['head']='1'
    return g

def basic_edges(g):
    """
    change edges {1:comp, 2:obl} to {1:comp}
    """
    def remove(e): return Fs_edge({'1': e['1']})
    for n in g.sucs:
        g.sucs[n] = tuple((m, remove(e)) for m, e in g.sucs[n])
    return g


def get_best_solution(corpus_gold, corpus_start, grs : GRS, strategy="main", verbose=0) -> CorpusDraft:
    """
    grs is a GRSDraft
    return the best solution using the grs with respect to the gold corpus
    the grs does not need to be confluent. We take the best solution with 
    respect to the f-score
    """
    def f_score(t):
        return t[0]/math.sqrt((t[0]+t[1])*(t[0]*t[2])+1e-20)

    solutions = grs.run(corpus_start)
    corpus = CorpusDraft(corpus_start)
    for sid in corpus_gold:
        best_fscore = 0
        for g in solutions[sid]:
            fs = f_score(g.edge_diff_up_to(corpus_gold[sid]))
            if fs > best_fscore:
                best_fscore = fs
                corpus[sid] = g
    return corpus

def diff_by_edge_value(corpus_gold, corpus):
    def filter(E,e):
        return {(m,n) for m,f,n in E if f == e}
    res = dict()
    for sid in corpus_gold:
        G1 = corpus_gold[sid]
        G2 = corpus[sid]
        E1 = set(G1.triples())
        E2 = set(G2.triples())
        edges = {e for (m,e,n) in E1} | {e for (m,e,n) in E2}
        res = res | {e : (0,0,0) for e in edges - set(res.keys())}
        for e in edges:
            c,l,r = res[e]
            E1e = filter(E1,e)
            E2e = filter(E2,e)
            c += len(E1e & E2e)
            l += len(E1e - E2e)
            r += len(E2e - E1e)
            res[e] = (c,l,r)
    return res

def prepare_corpus(filename):
    corpus = Corpus(CorpusDraft(filename).apply(append_head))
    empty = Corpus(CorpusDraft(corpus).apply(clear_but_working))
    return corpus, empty

def append_delete_head(grs : GRSDraft):
    for rn in grs:
        grs[rn].commands.append(Delete_feature("Y", "head"))
    return grs


def zero_knowledge_learning(gold, corpus_empty, request, args, param):
    nodes = request.named_entities()['nodes']
    draft, X, y, edge_idx, nkv_idx = observations(gold, request, nodes, param)
    idx2nkv = {v:k for k,v in nkv_idx.items()}
    rules = []
    for dependency in edge_idx:
        if dependency != None:
            dep = edge_idx[dependency]
            requests, _ = clf_dependency(dep,X,y,idx2nkv,request,('X','Y'),param,args.depth,False)
            rules += [Rule(Request(req).pattern('Y[head="1"]'),Commands(Add_edge('X',dependency,'Y'))) for req in requests]
    
    named_rules = Package({f'r{i}' : rules[i] for i in range(len(rules))})
    rules_with_head = append_delete_head(named_rules)
    grsd = GRSDraft({'Simple' : Package(rules_with_head), 'main' : 'Onf(Simple)'})
    if args.rules:
        grsd.save(args.rules)
    grs = GRS(grsd)
    print("testing")
    currently_computed_corpus = get_best_solution(corpus_gold, corpus_empty, grs, args.verbose)
    print(currently_computed_corpus.edge_diff_up_to(corpus_gold))
    print(diff_by_edge_value(currently_computed_corpus, draft))
    if args.web:
        web = grew_web.Grew_web()
        print(web.url())
        web.load_corpus(corpus_empty)
        web.load_grs(grs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='learner.py',
                                     description='Learn a grammar from sample files',
                                     epilog='Use connll files')
    parser.add_argument('train')
    parser.add_argument('-e', '--eval', default=None)
    parser.add_argument('-t', '--threshold', default=1e-1, type=float, help="minimal threshold to consider a node as pure")
    parser.add_argument('-d', '--depth', default=12, help="depth of the binary decision tree")
    parser.add_argument('--rules', default=None)
    parser.add_argument('-s','--nodetails',action="store_true")
    parser.add_argument('--verbose',default=0)
    parser.add_argument('-c', '--local_rules',action="store_true")
    parser.add_argument('-w','--web',action='store_true')
    args = parser.parse_args()
    if not args.eval:
        args.eval = args.train

    set_config("sud")
    param = {
        "min_samples_leaf": 5,
        "skipped_features":  {'xpos', 'SpaceAfter', 'Shared', 'textform', 'Typo', 'form', 'wordform', 'CorrectForm'},
        "node_impurity": 0.15,
        "threshold" : args.threshold,
        "tree_depth" : args.depth,
        "ratio" : 10
    }
    corpus_gold, corpus_empty = prepare_corpus(args.train)
    if args.nodetails:
        corpus_gold = Corpus(CorpusDraft(corpus_gold).apply(basic_edges))
    
    A = corpus_gold.count(Request('pattern{X<Y;e:X->Y}'))
    A += corpus_gold.count(Request('pattern{Y<X;e:X->Y}'))
    print("---target----")
    print(f"""number of edges within corpus: {corpus_gold.count(Request('pattern{e: X -> Y}'))}""")
    print(f"number of adjacent relations: {A}")
    print(corpus_gold.count(Request("pattern{e:X->Y}"), ["e.label"]))
    zero_knowledge_learning(corpus_gold, corpus_empty, Request('pattern{X[];Y[]}'),args, param)


"""

def edge_is(e,f):
    '''
    tells whether e is an f
    '''
    for k in e:
        if k not in f or e[k] not in f[k]:
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
def node_to_rule(n : int, e , back, T, request : Request, idx2nkv, idx2e, param, known_nodes):
    while n: #find the shortest path leading to e with impurity lower than threshold, we start with a leaf, so we go to the root
        father = back[n][1]
        ef = idx2e[np.argmax(T.value [ father ] )] #prevalent edge of the father
        if ef is None or ef != e:
            break
        if T.impurity[father] > param['node_impurity']:
            break
        n = father
    if not n:
        return []
    if n in known_nodes: #this node has been seen in the past, no rules is synthesized
        return []
    known_nodes.add(n)
    request = Request(request)  # builds a copy 
    while n != 0: #0 is the root node
        right, n = back[n]
        Z = idx2nkv[T.feature[n]]
        if isinstance(Z, tuple): #it is a feature pattern
            m, feat, feat_value = Z
            feat_value = feat_value.replace('"', '\\"')
            Z = f'{m}[{feat}="{feat_value}"]'
        if right:
            request.append("pattern", Z)
        else:
            request.without(Z)                    
    rule = Rule(request, Commands(Add_edge("X", e, "Y")))
    return [rule]

def decision_tree_to_rules(T, idx2e, idx2nkv, request, param):
    back, leaves = back_tree(T)
    known_nodes = set()
    res = []
    for n in leaves:
        e = idx2e[np.argmax( T.value[n])]
        if e and T.impurity[n] < param['node_impurity']:
            res += node_to_rule(n, e, back, T, request, idx2nkv, idx2e, param, known_nodes)
    return res

def forbidden_patterns(T, idx2e, idx2nkv, request, param):
    back, leaves = back_tree(T)
    internal_node = set()
    empty_patterns = []
    for n in leaves:
        if not idx2e[np.argmax(T.value[n])]:
            while n and back[n][1] and T.impurity[ back[n][1] ] <= 0.0001:
                n = back[n][1]
            if n and n not in internal_node and T.impurity[ n ] <= 0.0001:
                internal_node.add(n)
                req = Request(request)  # builds a copy 
                while n != 0: #0 is the root node
                    right, n = back[n]
                    Z = idx2nkv[T.feature[n]]
                    if isinstance(Z, tuple): #it is a feature pattern
                        m, feat, feat_value = Z
                        feat_value = feat_value.replace('"', '\\"')
                        Z = f'{m}[{feat}="{feat_value}"]'
                    if right:
                        req.append("pattern", Z)
                    else:
                        req.without(Z)  
                empty_patterns.append( req)
    return empty_patterns

    def pack(s):
    if re.search("f[XYZ]", s):
        return 3
    if re.search("intermediate", s):
        return 2
    if "enhanced" in s:
        return 1
    return 0

def add_rules(draft, xupos, yupos, edges, edge_idx, skipped_features):
    X,y,W,nkv_idx = build_Xy_by_grew(corpus_gold, draft, Request(f"X[upos={xupos}];Y[upos={yupos}]"),skipped_features, edge_idx) #
    clf = DecisionTreeClassifier(criterion="gini", 
                                 min_samples_leaf=param["min_samples_leaf"], 
                                 max_depth=8,
                                 class_weight={ edge_idx[e] : edges[e] for e in edges})
    print("learning")
    clf.fit(X, y, sample_weight=W)
    idx2nkv = {v:k for k,v in nkv_idx.items()}
    idx2e = {v:k for k,v in edge_idx.items()}
    return decision_tree_to_rules(clf.tree_, idx2e, idx2nkv, Request(f"X[upos={xupos}];Y[upos={yupos},head]"), param)
    

def standard_learning_process(corpus_gold, corpus_empty, args, param):
    R0,L0 = adjacent_rules(corpus_gold, param)
    print(len(L0))
    R0e = refine_rules(R0, corpus_gold, param)
    R0e = R0e.safe_rules()
    add_rule_name(R0e)
    R0e = append_delete_head(R0e)
    print(f"number of rules len(R0e) = {len(R0e)}")
    #turn R0e to a set of packages to speed up the process
    packages = {f'P{i}' : Package() for i in range(4)}
    for rn in R0e:
        packages[f'P{pack(rn)}'][rn] = R0e[rn]
    R00 = GRSDraft(packages)
    R00['main'] = "Seq(Onf(P0),Onf(P1),Onf(P2),Onf(P3))"
    if args.rules:
        R00.save(args.rules)

    G00 = GRS(R00)
    if args.web:
        web = grew_web.Grew_web()
        print(web.url())
        web.load_corpus(corpus_empty)
        web.load_grs(G00)

    currently_computed_corpus = get_best_solution(corpus_gold, corpus_empty, G00, args.verbose)
    print(currently_computed_corpus.edge_diff_up_to(corpus_gold))

    if args.local_rules:
        loc_rules,_ = local_rules(corpus_gold, param)
        Rloc = refine_rules(loc_rules, corpus_gold, param)
        Rloc = Rloc.safe_rules()
        Rloc = append_delete_head(Rloc)
        print(f"number of local rules: {len(Rloc)}")
        packages['P4'] = Package(Rloc)
        R00 = GRSDraft(packages)
        R00['main'] = "Seq(Onf(P0),Onf(P1),Onf(P2),Onf(P3),Onf(P4))"
        currently_computed_corpus = get_best_solution(corpus_gold, currently_computed_corpus, G00, strategy="Onf(P4)")
        print(currently_computed_corpus.edge_diff_up_to(corpus_gold))
        if args.rules:
            f = open(args.rules, "w")
            f.write(str(R00))
            f.close()
        print('done')

"""
