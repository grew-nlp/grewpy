import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

import grewpy
from grewpy import Corpus, GRS
from grewpy import Request, Rule, Commands, Command, GRSDraft, Graph, CorpusDraft
import numpy as np

#type declaration
Count = dict[str,int]
Observation = dict[str,dict[str,Count]]
#Observation is a dict mapping'VERB' to a dict mapping 'NOUN' to {'' : 10, 'xcomp': 7, 'obj': 36, 'nsubj': 4 ....}
#'' meaning no relationships
    
grewpy.set_config("sud")


def print_request_counter():
    print(f"Req: {grewpy.network.request_counter}")

def build_grs(rules : dict):
    """
    rules is a dict of Rules supposed to contain one Command of type add_edge
    we add the without clause to Rules and we add a main strategy
    """
    grs = dict()
    for rule_name, rule in rules.items():
        cde = rule.commands[0]
        safe_rule = Rule( rule.request.without(cde.safe()), rule.commands)
        grs[rule_name] = safe_rule
    grs["main"] = f"Onf(Alt({','.join(rule_name for rule_name in rules)}))"
    return GRS(grs)

def cluster(c : Corpus, P : Request, n1 : str,n2 : str) -> Observation:
    """
    search for P within c
    n1 and n2 are nodes within P
    """
    P1 = Request(P, f'e:{n1} -> {n2}')
    obs = c.count(P1, [f"{n1}.upos", f"{n2}.upos", "e.label"])
    W1 = Request(f"{n1}[];{n2}[]",P).without(f"{n1} -> {n2}")
    clus = c.count(W1, [f"{n1}.upos", f"{n2}.upos"])
    for u1 in obs:
        for u2 in obs[u1]:
            obs[u1][u2][''] = clus.get(u1,dict()).get(u2,0)
    return obs

def anomaly(obs : Count, threshold : float):
    s = sum(obs.values()) 
    for x in obs:
        if obs[x] > threshold * s and x:
            return x, obs[x]/s
    return None, None

def build_rules(requirement, rules, corpus, n1, n2, rule_name, rule_eval, threshold):
    """
    build rules corresponding to request requirement with help of corpus
    n1, n2 two nodes on which we do a cluster
    """
    obslr = cluster(corpus, requirement, n1, n2)
    for p1, v in obslr.items():
        for p2, es in v.items():
            (x, p) = anomaly(es, threshold) #the feature edge x has majority
            if x:
                #build the rule            
                P = Request(f"{n1}[upos={p1}]; {n2}[upos={p2}]", requirement)
                R = Rule(P, Commands( Command.add_edge(n1,x,n2)))
                rn = f"_{p1}_{rule_name}_{p2}_"
                rules[rn] = R
                rule_eval[rn] = (x,p)

def rank0(corpus : Corpus, param) -> dict[str,Rule]:
    """
    builds all rank 0 rules
    """
    rules = dict()
    rule_eval = dict()
    build_rules("X<Y", rules, corpus, "X", "Y", "lr", rule_eval, param["base_threshold"])
    build_rules("Y<X", rules, corpus, "X", "Y", "rl", rule_eval, param["base_threshold"])
    return rules, rule_eval

def edge_verification(g: Graph, h : Graph) -> np.array : 
    E1 = g.triples() #set of edges as triples
    E2 = h.triples()
    return np.array([len(E1 & E2), len(E1 - E2), len(E2 - E1)])

def verify(corpus_test, corpus_gold):
    """
    given two corpora, outputs the number of common edges, only left ones and only right ones
    """
    (common, left, right) = np.sum([edge_verification(corpus_test[sid],corpus_gold[sid]) for sid in corpus_test], axis=0)
    precision = common / (common + left)
    recall = common / (common + right)
    f_measure = 2*precision*recall / (precision+recall)
    return { 
        "common": common, 
        "left": left, 
        "right": right, 
        "precision": round (precision, 3),
        "recall": round (recall, 3),
        "f_measure": round(f_measure, 3),
    }


def nofilter(k,v,param):
    """
    return True if k=feature,v=feature value can be used for classification
    """
    if len(v) > param["feat_value_size_limit"] or len(v) == 1:
        return False
    if k in param["skip_features"]:
        return False
    return True

def get_tree(X,y,param):
    if not X or not len(X[0]):
        return None
    if max(y) == 0:
        return None
    #X_train, X_test, y_train, y_test = train_test_split(X, y)    
    clf = DecisionTreeClassifier(max_depth=param["max_depth"], max_leaf_nodes=max(y)+1, min_samples_leaf=param["min_samples_leaf"])
    clf.fit(X, y)
    return clf

def fvs(matchings, corpus, param):
    """
    return the set of (feature,values) of nodes X and Y in the matchings
    """
    features = {'X' : dict(), 'Y':dict()}
    for m in matchings:
        graph = corpus[m["sent_id"]]
        nodes = m['matching']['nodes']
        for n in nodes:
            N = graph[nodes[n]] #feature structure of N
            for k,v in N.items():
                if k not in features[n]:
                    features[n][k] = set()
                features[n][k].add(v)
    features = [(n,k,fv) for n in features for k,v in features[n].items() if nofilter(k,v,param) for fv in v]
    return {features[i]: i for i in range(len(features))}, features

def create_classifier(matchings, pos, corpus, param):
    X, y1,y = [], dict(), list()
    for m in matchings:
        graph = corpus[m["sent_id"]]
        nodes = m['matching']['nodes']
        obs = [0]*len(pos)
        for n in nodes:
            feat = graph[nodes[n]]
            for k, v in feat.items():
                if (n, k, v) in pos:
                    obs[pos[(n, k, v)]] = 1
        e = graph.edge(nodes['X'], nodes['Y'])
        if e not in y1:
            y1[e] = len(y1)
        y.append(y1[e])
        X.append(obs)
    
    return get_tree(X,y,param), {y1[i] : i for i in y1}


def find_classes(clf):
    def branches(pos, tree, current, acc, threshold):
        if tree.feature[pos] >= 0:
            #there is a feature
            if tree.children_left[pos] >= 0:
                #there is a child
                left = current + ((tree.feature[pos],1))
                branches(tree.children_left[pos], tree, left, acc, threshold)
            if tree.children_right[pos] >= 0:
                right = current + ((tree.feature[pos], 0))
                branches(tree.children_right[pos], tree, right, acc, threshold)
            return 
        else:
            if tree.impurity[pos] < threshold:
                acc[pos] = current
    tree = clf.tree_
    bs = dict()
    branches(0, tree, tuple(), bs, 0.01)
    return bs

def refine_rule(rule_name, R, corpus, n1, n2, param):
    res = []
    matchings = corpus.search(R.request.pattern())
    pos, fpat = fvs(matchings, corpus, param)
    clf, y1 = create_classifier(matchings, pos, corpus, param)
    if clf:
        branches = find_classes(clf)
        for node in branches:
            branch=branches[node]
            r = R.request.pattern()
            for i in range(0, len(branch), 2):
                n, feat, feat_value = fpat[branch[i]]
                if branch[i+1]:
                    r = r.without(f'{n}[{feat}="{feat_value}"]')
                else:
                    r.append("pattern",f'{n}[{feat}="{feat_value}"]')
            e = y1[ clf.tree_.value[node].argmax()]
            if e:
                r = r.without(f"{n1} -[{e}]-> {n2}")
                rule = Rule(r, Commands(f"add_edge {n1}-[{e}]-> {n2}"))
                res.append(rule)
    return res
    """
    import pickle
        tree.plot_tree(clf)
        #x = input("keep")
        if x:  
            pickle.dump(clf, open("hum.pickle", "wb"))
            x = 0
        """

def corpus_remove_edges(corpus):
    """
    create a corpus based on *corpus* whose graph have no edges
    """
    c = CorpusDraft(corpus)
    for sid,g in c.items():
        for n in g:
            g.sucs[n] = []
    return Corpus(c)

if __name__ == "__main__":
    param = {
        "base_threshold": 0.5,
        "valid_threshold": 0.9,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "max_leaf_node": 5,  # unused see DecisionTreeClassifier.max_leaf_node
        "feat_value_size_limit": 10,
        "skip_features": ['xpos', 'upos', 'SpaceAfter'],
    }
    # corpus_gold = Corpus("examples/resources/fr_pud-ud-test.conllu")
    corpus_gold = Corpus("examples/resources/pud_10.conllu")
    R0, rule_eval = rank0(corpus_gold, param)


    corpus_empty = corpus_remove_edges(corpus_gold)
    print(verify(corpus_empty, corpus_gold))
    print(f"len(R0) = {len(R0)}")
    Rs0 = build_grs(R0)
    
    corpus_rank0 = Corpus({ sid : Rs0.run(corpus_empty[sid], 'main')[0] for sid in corpus_empty})
    A = corpus_gold.count(Request("X[];Y[];X<Y;X->Y"),[])
    A += corpus_gold.count(Request("X[];Y[];Y<X;X->Y"), [])
    print(f"A = {A}")
    print(verify(corpus_rank0, corpus_gold))

    print(f"len(R0) = {len(R0)}")
    new_rules = dict()
    for rule_name, R in R0.items():
        if rule_eval[rule_name][1] < param["valid_threshold"]:
            X,Y = ("X","Y") if "lr" in rule_name else ("Y","X")
            new_r = refine_rule(rule_name, R, corpus_gold, X, Y, param)
            if new_r and len(new_r) >= 2:
                cpt = 1
                
                print("--------------------------replace")
                print(R)
                
                for r in new_r:
                    
                    print("by : ")
                    print(r)
                    
                    new_rules[f"{rule_name}_enhanced{cpt}"] = r
                    cpt += 1
        else:
            new_rules[rule_name] = R

    print(f"len(new_rules) = {len(new_rules)}")
    Rse = build_grs(new_rules)

    corpus_rank0_refined = Corpus({sid: Rse.run(corpus_empty[sid], 'main')[0] for sid in corpus_empty})
    print(f"A = {A}")
    print(verify(corpus_rank0_refined, corpus_gold))
