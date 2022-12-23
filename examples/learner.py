import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

from grewpy import Corpus, GRS, set_config
from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, CorpusDraft

#type declaration
Count = dict[str,int]
Observation = dict[str,dict[str,Count]]
#Observation is a dict mapping'VERB' to a dict mapping 'NOUN' to {'' : 10, 'xcomp': 7, 'obj': 36, 'nsubj': 4 ....}
#'' meaning no relationships
    
def cluster(corpus : Corpus, P : Request) -> Observation:
    """
    search for a link X -> Y with respect to pattern P in the corpus
    we build a cluster depending on X.upos and Y.upos
    """
    P1 = Request(P, f'e:X-> Y')
    obs = corpus.count(P1, [f"X.upos", f"Y.upos", "e.label"])
    W1 = Request(P).without(f"X -> Y")
    clus = corpus.count(W1, [f"X.upos", f"Y.upos"])
    for u1 in obs:
        for u2 in obs[u1]:
            obs[u1][u2][''] = clus.get(u1,dict()).get(u2,0)
    return obs

def anomaly(obs : Count, param):
    """
    return an observation and its probability if beyond base_threshold
    """
    s = sum(obs.values()) 
    for x in obs:
        if obs[x] > param["base_threshold"] * s and x:
            return x, obs[x]/s
    return None, None

def build_rules(base_pattern, rules : GRSDraft, corpus : Corpus, rule_name, rule_eval, param):
    """
    search a rule adding an edge X -> Y, given a base_pattern  
    we build the clusters, then
    for each pair (X, upos=U1), (Y, upos=U2), we search for 
    some edge e occuring at least with probability base_threshold
    in which case, we define a rule R: 
    base_pattern /\ [X.upos=U1] /\ [Y.upos=U2] => add_edge X-[e]-Y
    """
    obslr = cluster(corpus, base_pattern)
    for p1, v in obslr.items():
        for p2, es in v.items():
            (x, p) = anomaly(es, param) #the feature edge x has majority
            if x:
                #build the rule            
                P = Request(f"X[upos={p1}]; Y[upos={p2}]", base_pattern)
                c = Add_edge("X",x,"Y")
                R = Rule(P, Commands( c))
                rn = f"_{p1}_{rule_name}_{p2}_"
                rules[rn] = R
                rule_eval[rn] = (x,p)

def rank0(corpus : Corpus, param) -> GRSDraft:
    """
    build all rank 0 rules. They are supposed to connect words at distance 1
    """
    rules = GRSDraft()
    rule_eval = dict() #eval of the rule, same keys as rules
    build_rules("X[];Y[];X<Y", rules, corpus, "lr", rule_eval, param)
    build_rules("X[];Y[];Y<X", rules, corpus, "rl", rule_eval, param)
    return rules, rule_eval

def nofilter(k,v,param):
    """
    avoid to use meaningless features/feature values
    return True if k=feature,v=feature value can be used for classification
    """
    if len(v) > param["feat_value_size_limit"] or len(v) == 1:
        return False
    if k in param["skip_features"]:
        return False
    return True

def get_tree(X,y,param):
    """
    build a decision tree based on observation X,y
    given hyperparameter within param
    """
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
    return the set of (Z,feature,values) of nodes Z=X, Y in the matchings
    within the corpus. param serves to filter meaningful features values
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
    return features

def create_classifier(matchings, pos, corpus, param):
    """
    builds a decision tree classifier
    matchings: the matching on the two nodes X, Y
    we compute the class mapping (feature values of X, feature values of Y) to the edge between X and Y
    pos: maps a triple (node=X or Y, feature=Gen, feature_value=Fem) to a column number
    corpus serves to get back graphs
    param contains hyperparameter
    """
    X, y1,y = list(), dict(), list()
    #X: set of input values, as a list of (0,1)
    #y: set of output values, an index associated to the edge e
    #the absence of an edge has its own index
    #y1: the mapping between an edge e to some index
    for m in matchings:
        #each matching will lead to an obs which is a list of 0/1 values 
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
    """
    given a decision tree, extract "interesting" branches
    the output is a dict mapping the node_index to its branch
    a branch is the list of intermediate constraints = (7,1,8,0,...)
    that is feature value 7 has without clause whereas feature 8 is positive 
    """
    def branches(pos, tree, current, acc, threshold):
        if tree.feature[pos] >= 0:
            if tree.impurity[pos] < threshold:
                acc[pos] = current
                return
            #there is a feature
            if tree.children_left[pos] >= 0:
                #there is a child
                left = current + ((tree.feature[pos],1),)
                branches(tree.children_left[pos], tree, left, acc, threshold)
            if tree.children_right[pos] >= 0:
                right = current + ((tree.feature[pos], 0),)
                branches(tree.children_right[pos], tree, right, acc, threshold)
            return 
        else:
            if tree.impurity[pos] < threshold:
                acc[pos] = current
    tree = clf.tree_
    acc = dict()
    branches(0, tree, tuple(), acc, param["node_impurity"])
    return acc

def refine_rule(R, corpus, param):
    """
    takes a rule R, tries to find variants
    the result is the list of rules that should replace R
    for DEBUG, we return the decision tree classifier
    """
    res = []
    matchings = corpus.search(R.request)
    fpat = fvs(matchings, corpus, param) #the list of all feature values 
    pos = {fpat[i]: i for i in range(len(fpat))}
    clf, y1 = create_classifier(matchings, pos, corpus, param)
    if clf:
        branches = find_classes(clf, param) #extract interesting branches
        for node in branches:
            branch=branches[node]
            request = Request(R.request) #builds a new Request
            for feature_index, negative in branch:
                n, feat, feat_value = fpat[feature_index]
                if negative:
                    request = request.without(f'{n}[{feat}="{feat_value}"]')
                else:
                    request.append("pattern",f'{n}[{feat}="{feat_value}"]')
            e = y1[ clf.tree_.value[node].argmax()]
            if e: #here, e == None if there is no edges X -> Y
                rule = Rule(request, Commands(Add_edge("X",e,"Y")))
                res.append(rule)
    return res, clf
    """
    import pickle
        tree.plot_tree(clf)
        #x = input("keep")
        if x:  
            pickle.dump(clf, open("hum.pickle", "wb"))
            x = 0
        """

def refine_rules(Rs, corpus, param):
    """
    as above, but applies on a list of rules
    return the list of refined version of rules Rs
    """
    Rse = GRSDraft()
    for rule_name in Rs.rules():
        R = Rs[rule_name]
        if rule_eval[rule_name][1] < param["valid_threshold"]:
            new_r, clf = refine_rule(R, corpus, param)
            if len(new_r) >= 1:
                cpt = 1
                print("--------------------------replace")
                print(R)
                for r in new_r:
                    print("by : ")
                    print(r)
                    Rse[f"{rule_name}_enhanced{cpt}"] = r
                    cpt += 1

        else:
            Rse[rule_name] = R
    return Rse


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
        "base_threshold": 0.25,
        "valid_threshold": 0.90,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "max_leaf_node": 5,  # unused see DecisionTreeClassifier.max_leaf_node
        "feat_value_size_limit": 10,
        "skip_features": ['xpos', 'upos', 'SpaceAfter'],
        "node_impurity" : 0.1,
        "number_of_extra_leaves" : 3
    }
    corpus_gold = Corpus("examples/resources/fr_pud-ud-test.conllu")
    #corpus_gold = Corpus("examples/resources/pud_10.conllu")
    R0, rule_eval = rank0(corpus_gold, param)

    corpus_empty = corpus_remove_edges(corpus_gold)
    print(corpus_empty.diff(corpus_gold))
    print(f"len(R0) = {len(R0)}")
    R0_test = GRS(R0.safe_rules().onf()) 
  
    corpus_rank0 = Corpus({ sid : R0_test.run(corpus_empty[sid], 'main')[0] for sid in corpus_empty})
    A = corpus_gold.count(Request("X[];Y[];X<Y;X->Y"),[])
    A += corpus_gold.count(Request("X[];Y[];Y<X;X->Y"), [])
    print(f"A = {A}")
    print(corpus_rank0.diff(corpus_gold))

    print(f"len(R0) = {len(R0)}")
    R0e = refine_rules(R0, corpus_gold, param)

    print(f"len(new_rules) = {len(R0e)}")
    R0e_test = GRS(R0e.safe_rules().onf())

    corpus_rank0_refined = Corpus({sid: R0e_test.run(corpus_empty[sid], 'main')[0] for sid in corpus_empty})
    print(f"A = {A}")
    print(corpus_rank0_refined.diff(corpus_gold))
