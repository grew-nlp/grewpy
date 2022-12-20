import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

from grewpy import Corpus, GRS, set_config
from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, Graph, CorpusDraft

#type declaration
Count = dict[str,int]
Observation = dict[str,dict[str,Count]]
#Observation is a dict mapping'VERB' to a dict mapping 'NOUN' to {'' : 10, 'xcomp': 7, 'obj': 36, 'nsubj': 4 ....}
#'' meaning no relationships
    
def cluster(corpus : Corpus, P : Request, n1 : str,n2 : str) -> Observation:
    """
    search for P within c
    n1 and n2 are nodes within P
    """
    P1 = Request(P, f'e:{n1} -> {n2}')
    obs = corpus.count(P1, [f"{n1}.upos", f"{n2}.upos", "e.label"])
    W1 = Request(f"{n1}[];{n2}[]",P).without(f"{n1} -> {n2}")
    clus = corpus.count(W1, [f"{n1}.upos", f"{n2}.upos"])
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

def build_rules(base_pattern, rules : GRSDraft, corpus, n1, n2, rule_name, rule_eval, threshold):
    """
    build rules corresponding to request base_pattern with help of corpus
    n1, n2 two nodes on which we do a cluster
    """
    obslr = cluster(corpus, base_pattern, n1, n2)
    for p1, v in obslr.items():
        for p2, es in v.items():
            (x, p) = anomaly(es, threshold) #the feature edge x has majority
            if x:
                #build the rule            
                P = Request(f"{n1}[upos={p1}]; {n2}[upos={p2}]", base_pattern)
                c = Add_edge(n1,x,n2)
                R = Rule(P, Commands( c))
                rn = f"_{p1}_{rule_name}_{p2}_"
                rules[rn] = R
                rule_eval[rn] = (x,p)

def rank0(corpus : Corpus, param) -> GRSDraft:
    """
    build all rank 0 rules
    """
    rules = GRSDraft()
    rule_eval = dict() #eval of the rule, same keys as rules
    build_rules("X<Y", rules, corpus, "X", "Y", "lr", rule_eval, param["base_threshold"])
    build_rules("Y<X", rules, corpus, "X", "Y", "rl", rule_eval, param["base_threshold"])
    return rules, rule_eval

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
    clf = DecisionTreeClassifier(max_depth=param["max_depth"], 
                                max_leaf_nodes=max(y)+param["number_of_extra_leaves"],
                                min_samples_leaf=param["min_samples_leaf"], 
                                criterion="gini")
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
    X, y1,y = list(), dict(), list()
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


def find_classes(clf, param):
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
    branches(0, tree, tuple(), bs, param["node_impurity"])
    return bs

def refine_rule(rule_name, R, corpus, n1, n2, param):
    res = []
    matchings = corpus.search(R.request.pattern())
    pos, fpat = fvs(matchings, corpus, param)
    clf, y1 = create_classifier(matchings, pos, corpus, param)
    if clf:
        branches = find_classes(clf, param)
        for node in branches:
            branch=branches[node]
            request = Request(R.request)
            for i in range(0, len(branch), 2):
                n, feat, feat_value = fpat[branch[i]]
                if branch[i+1]:
                    request = request.without(f'{n}[{feat}="{feat_value}"]')
                else:
                    request.append("pattern",f'{n}[{feat}="{feat_value}"]')
            e = y1[ clf.tree_.value[node].argmax()]
            if e:
                rule = Rule(request, Commands(Add_edge(n1,e,n2)))
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
    create a corpus based on *corpus* whose graphs have no edges
    """
    def clear_edges(g):
        for n in g:
            g.sucs[n] = []
        return g
    return Corpus(CorpusDraft(corpus).map(clear_edges))

if __name__ == "__main__":
    set_config("sud")
    param = {
        "base_threshold": 0.5,
        "valid_threshold": 0.95,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "max_leaf_node": 5,  # unused see DecisionTreeClassifier.max_leaf_node
        "feat_value_size_limit": 10,
        "skip_features": ['xpos', 'upos', 'SpaceAfter'],
        "node_impurity" : 0.001,
        "number_of_extra_leaves" : 2
    }
    corpus_gold = Corpus("examples/resources/fr_pud-ud-test.conllu")
    #corpus_gold = Corpus("examples/resources/pud_10.conllu")
    R0, rule_eval = rank0(corpus_gold, param)

    corpus_empty = corpus_remove_edges(corpus_gold)
    print(corpus_empty.diff(corpus_gold))
    print(f"len(R0) = {len(R0)}")
    R0.safe_rules()
    R0.onf() #add strategy Onf(Alt(rules))
    GR0 = GRS(R0)
  
    corpus_rank0 = Corpus({ sid : GR0.run(corpus_empty[sid], 'main')[0] for sid in corpus_empty})
    A = corpus_gold.count(Request("X[];Y[];X<Y;X->Y"),[])
    A += corpus_gold.count(Request("X[];Y[];Y<X;X->Y"), [])
    print(f"A = {A}")
    print(corpus_rank0.diff(corpus_gold))

    print(f"len(R0) = {len(R0)}")
    R0e = GRSDraft()
    for rule_name in R0.rules():
        R = R0[rule_name]
        if rule_eval[rule_name][1] < param["valid_threshold"]:
            X,Y = ("X","Y") if "lr" in rule_name else ("Y","X")
            new_r = refine_rule(rule_name, R, corpus_gold, X, Y, param)
            if len(new_r) >= 1:
                cpt = 1
                
                print("--------------------------replace")
                print(R)
                
                for r in new_r:                    
                    print("by : ")
                    print(r)
                    
                    R0e[f"{rule_name}_enhanced{cpt}"] = r
                    cpt += 1
        else:
            R0e[rule_name] = R

    print(f"len(new_rules) = {len(R0e)}")
    R0e.safe_rules()
    R0e.onf()
    GR0e = GRS(R0e)

    corpus_rank0_refined = Corpus({sid: GR0e.run(corpus_empty[sid], 'main')[0] for sid in corpus_empty})
    print(f"A = {A}")
    print(corpus_rank0_refined.diff(corpus_gold))
