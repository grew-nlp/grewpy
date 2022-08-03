from audioop import mul
import grew
from utils import multi_append

c = grew.Corpus("UD_French-PUD/fr_pud-ud-test.conllu")

def pair(c,matching,e,n1,n2):#append the matching in e
    G = c[matching['sent_id']]
    P1 = G[matching['matching']['nodes'][n1]].get("upos", "")
    P2 = G[matching['matching']['nodes'][n2]].get("upos", "")
    if P1 and P2:
        multi_append(e, (P1, P2), str(matching['matching']['edges']['e']['label']))

def cluster(c,P,n1,n2):
    obs = dict()
    for matching in c.search(P):
        pair(c,matching,obs,n1,n2)
    for u1,u2 in obs:
        W = grew.Pattern(
            ("pattern", [f"Y[upos={u2}]", f"X[upos={u1}]"]), 
            P[1],
            ("without", ["X -> Y"]))
        obs[(u1, u2)][""] = c.count(W)
    return obs

def anomaly(obs):
    s = sum(obs.values())
    for x in obs:
        if obs[x] > 0.95 * s and x:
            return x

def rank0(c):
    """
    builds all rank 0 rules
    """
    Plr = grew.Pattern(("pattern", ["X<Y"]), ("pattern", ["e:X -> Y"]))
    obslr = cluster(c, Plr, "X", "Y") #left to right
    Prl = grew.Pattern(("pattern", ["Y<X"]), ("pattern", ["e:X -> Y"]))
    obsrl = cluster(c, Prl, "X", "Y")
    rules = []
    for (p1,p2),es in obslr.items():
        if x := anomaly(es):
            #build the rule
            #print(f"{p1} -> {p2} : {x}, {es}")
            P = grew.Pattern(("pattern", ["X<Y", f"X[upos={p1}]", f"Y[upos={p2}]"]))
            R = grew.Rule(f"_{p1}_lr_{p2}_",P,grew.Command(f"add_edge X-[{x}]->Y"))
            rules.append(R)
    for (p1, p2), es in obsrl.items():
        if x := anomaly(es):
            #print(f"{p1} <- {p2} : {x} {es}")
            P = grew.Pattern(
                ("pattern", ["Y<X", f"X[upos={p1}]", f"Y[upos={p2}]"]))
            R = grew.Rule(f"_{p1}_rl_{p2}_",P,grew.Command(f"add_edge X-[{x}]->Y"))
            rules.append(R)
    return rules

R0 = rank0(c)
gcs = {sid : c[sid] for sid in c}#set of graphs
g0s = {sid : c[sid] for sid in c} #a copy of the graphs
for sid,g in g0s.items():
    for n in g:
        g.edges[n]=[] #trash all edges in g0s

def verify(gs,hs):
    recall, found = 0,0
    for sid,g in gs.items():
        for n in g:
            for e,s in g.edges[n]:
                if (e,s) in hs[sid].edges[n]:
                    found += 1
                else:
                    recall += 1
    return found, recall

print(verify(g0s,gcs))
    
GRS = grew.GRS()
GRS.packages["main"] = R0



"""
for (p1,p2),V in e12.items():
    print(f"{p1} -> {p2} : {V}")
for (p1, p2), V in e21.items():
    print(f"{p1} <- {p2} : {V}")
"""

"""
def filter_upos(g):
    for n in g:
        g[n] =  {"upos": g[n]["upos"]} if "upos" in g[n] else {"upos":""}
    return g

"""