from audioop import mul
from context import grewpy
import grew
from utils import multi_append

c = grew.Corpus("test.conllu") #UD_French-PUD/fr_pud-ud-test.conllu")

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

def pair(c,matching,e,n1,n2):#append the matching in e
    G = c[matching['sent_id']]
    P1 = G[matching['matching']['nodes'][n1]].get("upos", "")
    P2 = G[matching['matching']['nodes'][n2]].get("upos", "")
    if P1 and P2:
        multi_append(e, (P1, P2), parse_edge(matching['matching']['edges'],'e'))

def cluster(c,P,n1,n2):
    obs = dict()
    for matching in c.search(P):
        pair(c,matching,obs,n1,n2)
    for u1,u2 in obs:
        W = grew.Pattern(
            ("pattern", f"Y[upos={u2}];X[upos={u1}]"), 
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
    Plr = grew.Pattern(("pattern", "X<Y"), ("pattern", "e:X -> Y"))
    obslr = cluster(c, Plr, "X", "Y") #left to right
    Prl = grew.Pattern(("pattern", "Y<X"), ("pattern", "e:X -> Y"))
    obsrl = cluster(c, Prl, "X", "Y")
    rules = []
    for (p1,p2),es in obslr.items():
        if x := anomaly(es):
            #build the rule            
            P = grew.Pattern(
                ("pattern", ["X<Y", f"X[upos={p1}]", f"Y[upos={p2}]"]), ("without", f"X-[{x}]->Y"))
            R = grew.Rule(f"_{p1}_lr_{p2}_",P,grew.Command(f"add_edge X-[{x}]->Y"))
            rules.append(R)
    for (p1, p2), es in obsrl.items():
        if x := anomaly(es):
            #print(f"{p1} <- {p2} : {x} {es}")
            P = grew.Pattern(
                ("pattern", ["Y<X", f"X[upos={p1}]", f"Y[upos={p2}]"]),("without",f"X-[{x}]->Y"))
            R = grew.Rule(f"_{p1}_rl_{p2}_",P,grew.Command(f"add_edge X-[{x}]->Y"))
            rules.append(R)
    return rules

R0 = rank0(c)
gcs = {sid : c[sid] for sid in c}#set of graphs
g0s = {sid : c[sid] for sid in c} #a copy of the graphs
for sid,g in g0s.items():
    for n in g:
        g.sucs[n]=[] #trash all edges in g0s

def verify(gs,hs):
    recall, found = 0,0
    for sid,g in gs.items():
        for n in g:
            for e,s in g.sucs[n]:
                if (e,s) in hs[sid].sucs[n]:
                    found += 1
                else:
                    recall += 1
    return found, recall

print(verify(g0s,gcs))
print(len(R0))  
Rs0 = grew.GRS("rank0",rules=R0,
                       strats=[grew.Strategy('main',f'Onf(Alt({",".join([r.name for r in R0])}))')])


g1s = {
    sid : Rs0.run(g0s[sid], 'main')[0] 
    for sid in g0s}
print(verify(g1s,gcs))