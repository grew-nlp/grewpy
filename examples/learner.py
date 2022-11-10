import sys, os

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew
from utils import multi_append
from grew import Request, Command, Rule, Graph
from corpus import Corpus
import numpy as np

#type declaration
Count = dict[str,int]
Observation = dict[tuple[str,str],Count]
#Observation is a dict mapping('VERB', 'NOUN') to {'' : 10, 'xcomp': 7, 'obj': 36, 'nsubj': 4 ....}
#'' meaning no relationships

def parse_edge(m,e_name):
    """
    m is a matching
    """
    label = m[e_name]['label']
    if isinstance(label,str):
        return label
    elif isinstance(label,dict):
        if '1' in label and '2' in label:
            return f"1={label['1']},2={label['2']}"
    raise ValueError(m[e_name]['label'])

def pair(c : Corpus,matching,e,n1,n2):#append the matching in e
    G = c[matching['sent_id']]
    P1 = G[matching['matching']['nodes'][n1]].get("upos", "")
    P2 = G[matching['matching']['nodes'][n2]].get("upos", "")
    if P1 and P2:
        multi_append(e, (P1, P2), parse_edge(matching['matching']['edges'],'e'))

def cluster(c : Corpus, P : Request, n1 : str,n2 : str) -> Observation:
    """
    search for P within c
    n1 and n2 are nodes within P
    
    """
    obs = dict()
    P1 = Request(P, f'e:{n1} -> {n2}')
    for matching in c.search(P1):
        pair(c,matching,obs,n1,n2)
    for u1,u2 in obs: #take into account there is no edge between X and Y
        W = Request(P,f"Y[upos={u2}];X[upos={u1}]").without("X -> Y")
        obs[(u1, u2)][""] = c.count(W)
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
    for (p1,p2),es in obslr.items():
        if x := anomaly(es): #the feature edge x has majority
            #build the rule            
            P = Request(f"X[upos={p1}]; Y[upos={p2}]; X < Y").without( f"X-[{x}]->Y")
            R = Rule(P,Command(f"add_edge X-[{x}]->Y"))
            rules[f"_{p1}_lr_{p2}_"] = R
    for (p1, p2), es in obsrl.items():
        if x := anomaly(es):
            P = Request(f"X[upos={p1}], Y[upos={p2}], Y < X").without(f"X-[{x}]->Y")
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
    corpus = Corpus("examples/resources/fr_pud-ud-test.conllu")
    R0 = rank0(corpus)
    g0s = {sid : corpus[sid] for sid in corpus} #a copy of the graphs
    for sid,g in g0s.items():
        clear_edges(g)
    #cstart = Corpus(g0s)

    print(verify(g0s,corpus))
    print(len(R0))  
    Rs0 = grew.GRS(R0 | {'main' : f'Onf(Alt({",".join([r for r in R0])}))'})
    g1s = { sid : Rs0.run(g0s[sid], 'main')[0] for sid in g0s}
    print(verify(g1s,corpus))
