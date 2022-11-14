import sys, os

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

import grew
from grew import Corpus
from grew import Request, Rule, Command, GRS, Graph
import numpy as np

#type declaration
Count = dict[str,int]
Observation = dict[tuple[str,str],Count]
#Observation is a dict mapping('VERB', 'NOUN') to {'' : 10, 'xcomp': 7, 'obj': 36, 'nsubj': 4 ....}
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
    W1 = Request(f"Y[];X[]",P).without("X -> Y")
    clus = c.count(W1, [f"{n1}.upos", f"{n2}.upos"])
    for u1 in obs:
        for u2 in obs[u1]:
            obs[u1][u2][''] = clus.get(u1,dict()).get(u2,0)
    return obs

def anomaly(obs : Count):
    s = sum(obs.values()) 
    for x in obs:
        if obs[x] > 0.95 * s and x:
            return x

def rank0(c : Corpus) -> dict[str,Rule]:
    """
    builds all rank 0 rules
    """
    request_l_r = Request("X<Y")
    obslr = cluster(c, request_l_r, "X", "Y") #left to right
    request_r_l = Request("Y<X")
    obsrl = cluster(c, request_r_l, "X", "Y")
    rules = dict()
    for p1, v in obslr.items():
        for p2, es in v.items():
            if x := anomaly(es): #the feature edge x has majority
                #build the rule            
                P = Request(f"X[upos={p1}]; Y[upos={p2}]; X < Y").without( f"X-[{x}]->Y")
                R = Rule(P,Command(f"add_edge X-[{x}]->Y"))
                rules[f"_{p1}_lr_{p2}_"] = R
    for p1, v in obsrl.items():
        for p2, es in v.items():
            if x := anomaly(es):
                P = Request(f"X[upos={p1}]; Y[upos={p2}]; Y < X").without(f"X-[{x}]->Y")
                R = Rule(P,Command(f"add_edge X-[{x}]->Y"))
                rules[f"_{p1}_rl_{p2}_"] = R
    return rules

def edge_verification(g: Graph, h : Graph) -> tuple[int,int,int] : 
    E1 = set((nid, e, s) for nid in g for (e, s) in g.suc(nid))
    E2 = set((nid, e, s) for nid in h for (e, s) in h.suc(nid))
    return np.array([len(E1 & E2), len(E1 - E2), len(E2 - E1)])

def verify(gs, hs):
    """
    given two corpora, outputs the number of common edge, the only left ones and the only right ones
    """
    clr = np.zeros((3,),dtype=np.int32) #(0,0,0) in numpy
    for sid in gs:
        clr += edge_verification(gs[sid],hs[sid])
    return list(clr)

def clear_edges(g):
    for n in g:
        g.sucs[n] = []

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
    print(len(Rs0))

    g1s = { sid : Rs0.run(g0s[sid], 'main')[0] for sid in g0s}
    print_request_counter()
    print(verify(g1s, corpus))
    print_request_counter()
