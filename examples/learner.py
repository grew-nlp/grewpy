import math
import numpy as np
import argparse
import pickle
from typing import List, Union

from rule_forgery import observations, clf_dependency
# Use local grew lib
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from grewpy import Request, Rule, Commands, Command, Add_edge, Delete_feature
from grewpy import Corpus, GRS, set_config
from grewpy import GRSDraft, CorpusDraft, Package, Graph
from grewpy import grew_web
from grewpy.graph import Fs_edge

def add_rule_name(wgrs : List[Rule]):
    """
    add the name of the applied rule to node Y
    """
    for rn in wgrs: wgrs[rn].commands.append(Command(f'Y.rule="{rn}"'))

def clear_but_working(g : Graph):
    """
    delete non working edges within g
    """
    g.sucs = {n : [] for n in g.sucs}
    return (g)

def append_head(g : Graph):
    """
    each node in g get a head feature = 1
    """
    for n in g: g[n]['head']='1'
    return g

def basic_edges(g : Graph):
    """
    change edges {1:comp, 2:obl} to {1:comp}
    """
    def remove(e): return Fs_edge({'1': e['1']})
    for n in g.sucs:
        g.sucs[n] = tuple((m, remove(e)) for m, e in g.sucs[n])
    return g


def get_best_solution(corpus_gold, corpus_start, grs : GRS, strategy="main", verbose=0) -> CorpusDraft:
    """
    grs is a GRS
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

def clean_corpus(corpus : CorpusDraft, gold : Union[Corpus, CorpusDraft]):
    """
    remove edges from corpus that are not present in gold
    """
    for sid in corpus:
        for n in corpus[sid].sucs:
            corpus[sid].sucs[n] = [e for e in corpus[sid].sucs[n] if n in gold[sid].sucs and e in gold[sid].sucs[n]]

def diff_by_edge_value(corpus_gold, corpus) -> dict:
    """
    return for each edge value (e.g. subj) the difference between gold and corpus
    difference is given as a triple (common, only left, only right)
    """
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
    """
    builds two corpora : one for which each node is head (has no known father)
    and an empty corpus
    """
    corpus = Corpus(CorpusDraft(filename).apply(append_head))
    empty = Corpus(CorpusDraft(corpus).apply(clear_but_working))
    return corpus, empty

def append_delete_head(grs : GRSDraft):
    """
    When a node got its father, we remove its head feature
    """
    for rn in grs: grs[rn].commands.append(Delete_feature("Y", "head"))
    return grs


def zero_knowledge_learning(gold, corpus, request, args, param):
    """
    first compute rules, bundled within package, 
    then apply rules to corpus, 
    return the computed (cleaned with respect to gold) corpus and the package
    """
    named_entities = request.named_entities() #both nodes and edge names in the request
    gold_draft = CorpusDraft(corpus_gold)
    X, y, edge_idx, nkv_idx = observations(gold, gold_draft, request, named_entities, param)
    idx_edge = {v : k.compact() for (k,v) in edge_idx.items() if k != None}
    idx2nkv = {v:k for k,v in nkv_idx.items()}
    rules = []
    for dependency in edge_idx:
        if dependency != None and len(y)>0:
            dep = edge_idx[dependency]
            requests, _ = clf_dependency(dep,X,y,idx2nkv,request,named_entities, idx_edge, param,args.depth, False)
            rules += [Rule(Request(req).pattern('Y[head="1"]'),Commands(Add_edge('X',dependency,'Y'))) for req in requests]
    
    named_rules = Package({f'r{i}' : rules[i] for i in range(len(rules))})
    rules_with_head = append_delete_head(named_rules)
    package = Package(rules_with_head)
    grs = GRS({f'Simple' : package, 'main' : 'Onf(Simple)'})
    print("testing")
    currently_computed_corpus = get_best_solution(corpus_gold, corpus, grs, args.verbose)
    print(currently_computed_corpus.edge_diff_up_to(corpus_gold))
    print(diff_by_edge_value(currently_computed_corpus, gold_draft))
    #clean the input corpus
    clean_corpus(currently_computed_corpus, gold_draft)
    return currently_computed_corpus, package


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
        "ratio" : 20
    }
    corpus_gold, corpus_empty = prepare_corpus(args.train)
    if args.nodetails:
        corpus_gold = Corpus(CorpusDraft(corpus_gold).apply(basic_edges))
    
    #web = grew_web.Grew_web()
    #print(web.url())

    A = corpus_gold.count(Request('pattern{X<Y;e:X->Y}'))
    A += corpus_gold.count(Request('pattern{Y<X;e:X->Y}'))
    print("---target----")
    print(f"""number of edges within corpus: {corpus_gold.count(Request('pattern{e: X -> Y}'))}""")
    print(f"number of adjacent relations: {A}")
    print(corpus_gold.count(Request("pattern{e:X->Y}"), ["e.label"]))
    corpus1, package_1 = zero_knowledge_learning(corpus_gold, corpus_empty, Request('pattern{X[];Y[]}'),args, param)
    corpus2, package_2 = zero_knowledge_learning(corpus_gold, corpus1, Request('pattern{X[];Y[];Z[];f:X->Z}'),args, param)
    corpus3, package_3 = zero_knowledge_learning(corpus_gold, corpus2, Request('pattern{X[];Y[];Z[];f:Z->X}'),args, param)
    corpus4, package_4 = zero_knowledge_learning(corpus_gold, corpus3, Request('pattern{X[];Y[];Z[];f:Y->Z}'),args, param)
    corpus5, package_5 = zero_knowledge_learning(corpus_gold, corpus4, Request('pattern{X[];Y[];Z[];U[];f:Y->Z;g:X->U}'),args, param)
    corpus6, package_6 = zero_knowledge_learning(corpus_gold, corpus5, Request('pattern{X[];Y[];Z[];U[];f:Y->Z;g:U->X}'),args, param)
    pckags = [package_1, package_2, package_3, package_4, package_5, package_6]

    pkg_list = ','.join(f"Onf(P{i+1})" for i in range(6))
    grsd = GRSDraft({ f'P{i+1}' : pckags[i] for i in range(len(pckags))} | {'main' : f'Seq({pkg_list})'})
    grs = GRS(grsd)
    final_corpus = get_best_solution(corpus_gold, corpus6, grs, args.verbose)
    print(final_corpus.edge_diff_up_to(corpus_gold))
    print(diff_by_edge_value(final_corpus, corpus_gold))
    if args.rules:
        grsd.save(args.rules)
    if args.web:
        web = grew_web.Grew_web()
        print(web.url())
        web.load_corpus(corpus_empty)
        grs = GRS(grsd)
        web.load_grs(grs)
