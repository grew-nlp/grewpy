import grew

c = grew.Corpus("UD_French-PUD/fr_pud-ud-test.conllu")

def add_one(m,k):
    #m is a multiset, m[k] ++
    a = m.get(k,0)
    m[k] = a+1

def append(d, k, vs):
    """
    d map k to the multiset of v
    vs is either a list of v or a v
    append value v to d[k]
    """
    if k not in d:
        d[k] = dict()
    if isinstance(vs,list):
        for v in vs:
            add_one(d[k], v)
    else:
        add_one(d[k],vs)

def pair(c,matching,e,n1,n2):#append the matching in e
    G = c[matching['sent_id']]
    P1 = G[matching['matching']['nodes'][n1]].get("upos", "")
    P2 = G[matching['matching']['nodes'][n2]].get("upos", "")
    if P1 and P2:
        append(e, (P1, P2), str(matching['matching']['edges']['e']['label']))

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


def pairs_of_upos(c):
    """
    build the set of pairs of consecutive upos within c
    """
    P = grew.Pattern(("pattern", ["e:X -> Y"]),("pattern",["X<Y"]))
    Q = grew.Pattern(("pattern", ["e:X -> Y"]), ("pattern",["Y<X"]))
    e12 = cluster(c, P, "X", "Y")
    e21 = cluster(c,Q, "X","Y") 
    return e12,e21

e12, e21 = pairs_of_upos(c)
for (p1,p2),V in e12.items():
    print(f"{p1} -> {p2} : {V}")
for (p1, p2), V in e21.items():
    print(f"{p1} <- {p2} : {V}")

"""
def filter_upos(g):
    for n in g:
        g[n] =  {"upos": g[n]["upos"]} if "upos" in g[n] else {"upos":""}
    return g

"""
