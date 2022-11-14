import sys, os

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

import grew
from grew import Corpus
from grew import Request, Rule, Command, GRS, Graph
import numpy as np

#type declaration
Count = dict[str,int]
Observation = dict[str,dict[str,Count]]
#Observation is a dict mapping'VERB' to a dict mapping 'NOUN' to {'' : 10, 'xcomp': 7, 'obj': 36, 'nsubj': 4 ....}
#'' meaning no relationships
    
def print_request_counter():
    print(f"Req: {grew.network.request_counter}")

def cluster(c : Corpus, P : Request, n1 : str,n2 : str) -> Observation:
    """
    search for P within c
    n1 and n2 are nodes within P
    """
    P1 = Request(P, f'e:{n1} -> {n2}')
    obs = c.count(P1, [f"{n1}.upos", f"{n2}.upos", "e.label"])
    W1 = Request(f"{n1}[];{n2}[]",P).without("X -> Y")
    clus = c.count(W1, [f"{n1}.upos", f"{n2}.upos"])
    for u1 in obs:
        for u2 in obs[u1]:
            obs[u1][u2][''] = clus.get(u1,dict()).get(u2,0)
    return obs

def anomaly(obs : Count, threshold : float):
    s = sum(obs.values()) 
    for x in obs:
        if obs[x] > threshold * s and x:
            return x

def build_rules(requirement, rules, corpus, n1, n2, rule_name):
    """
    build rules corresponding to request requirement with help of corpus
    n1, n2 two nodes on which we do a cluster
    """
    obslr = cluster(corpus, requirement, n1, n2)
    for p1, v in obslr.items():
        for p2, es in v.items():
            if x := anomaly(es, 0.95): #the feature edge x has majority
                #build the rule            
                P = Request(f"{n1}[upos={p1}]; {n2}[upos={p2}]", requirement).without( f"{n1}-[{x}]->{n2}")
                R = Rule(P,Command(f"add_edge {n1}-[{x}]->{n2}"))
                rules[f"_{p1}_{rule_name}_{p2}_"] = R

def rank0(c : Corpus) -> dict[str,Rule]:
    """
    builds all rank 0 rules
    """
    rules = dict()
    build_rules("X<Y", rules, corpus, "X", "Y", "lr")
    build_rules("Y<X", rules, corpus, "X", "Y", "rl")
    return rules

def edge_verification(g: Graph, h : Graph) -> np.array : 
    E1 = g.edges_as_triple()
    E2 = h.edges_as_triple()
    return np.array([len(E1 & E2), len(E1 - E2), len(E2 - E1)])

def verify(corpus1, corpus2):
    """
    given two corpora, outputs the number of common edges, only left ones and only right ones
    """
    return list(np.sum([edge_verification(corpus1[sid],corpus2[sid]) for sid in corpus1], axis=0))

def clear_edges(graph):
    for n in graph:
        graph.sucs[n] = []

if __name__ == "__main__":
    print_request_counter()
    corpus = Corpus("examples/resources/fr_pud-ud-test.conllu")
    print_request_counter()
    R0 = rank0(corpus)
    print_request_counter()
    g0s = {sid: Graph(corpus[sid]) for sid in corpus}  # a copy of the graphs
    for sid,g in g0s.items():
        clear_edges(g)
    #cstart = Corpus(g0s)

    print_request_counter()
    print(verify(g0s, corpus))
    print_request_counter()
    print(len(R0))
    print_request_counter()
    Rs0 = GRS(R0 | {'main': f'Onf(Alt({",".join([r for r in R0])}))'})
    
    g1s = { sid : Rs0.run(g0s[sid], 'main')[0] for sid in g0s}
    print_request_counter()
    print(verify(g1s, corpus))
    print_request_counter()
