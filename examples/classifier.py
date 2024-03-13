from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np
# Use local grew lib
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")))

from grewpy.sketch import Sketch
from grewpy.observation import Observation
from grewpy import Corpus, GRS, set_config
from grewpy import Request, Rule, Commands, Add_edge, GRSDraft, CorpusDraft

def back_tree(T):
    """
    Given a classifier tree T, builds the parent relation: Y |-> (i,X) for 
    if Y is the i-th son of X in T
    returns also the set of leaves of T
    """
    back = dict()
    for node in range(T.node_count):
        g = T.children_left[node]
        d = T.children_right[node]
        if g >= 0: back[g] = (0, node)
        if d >= 0: back[d] = (1, node)
    return back

def branch(n, back):
    """
    starting from n, compute the branch from root node 0 to n
    by means of the back relation
    """
    branch = []
    while n != 0: #0 is the root node
        right, n = back[n] #right = is n the right son of its father?
        branch.append((n, right))
    branch.reverse() #get the branch in correct order
    return branch

def e_index(d):
    """
    given a collection d, maps an index to each element in d
    """
    cpt = iter(range(10000000))
    return {e : next(cpt) for e in d}


'''
def feature_values_for_decision(matchings, corpus, param, nodes):
    """
    restrict feature values to those useful for a decision tree
    """
    def feature_value_occurences(matchings, corpus):
        """
    given a matchings corresponding to some request on the corpus,
    return a dict mapping (n,feature) =>(values)=>occurrences to its occurence number in matchings
    within the corpus. n : node name, feature like 'Gender', values like 'Fem'
        """
        observation = Observation()
        for m in matchings:
            graph = corpus[m["sent_id"]]
            nodes = m['matching']['nodes']
            for n in nodes:
                N = graph[nodes[n]]  # feature structure of N
                for k, v in N.items():
                    if (n, k) not in observation:
                        observation[(n, k)] = dict()
                    observation[(n, k)][v] = observation[(n, k)].get(v, 0)+1
        return observation

    observation = feature_value_occurences(matchings, corpus)
    features = dict()
    for (n,k) in observation:
        if n in nodes and k not in param["skip_features"]:
            for v in observation.zipf(n, k, param["feat_value_size_limit"], param["zipf_feature_criterion"]):
                features[(n,k,v)] = observation[(n,k)][v]
    return features


def node_to_rule(n : int, e , back, T, request : Request, idx2nkv, param, internal_node):
    while n and back[n][1] and T.impurity[back[n][1]] < param['node_impurity']:
        n = back[n][1]
    if n in internal_node:
        return None
    if not n:
        return None
    request = Request(request)  # builds a copy 
    while n != 0: #0 is the root node
        right, n = back[n]
        Z = idx2nkv[T.feature[n]]
        if isinstance(Z, tuple): #it is a feature pattern
            m, feat, feat_value = Z
            feat_value = feat_value.replace('"', '\\"')
            Z = f'{m}[{feat}="{feat_value}"]'
        if right:
            request.append("pattern", Z)
        else:
            request.without(Z)                    
    rule = Rule(request, Commands(Add_edge("X", e, "Y")))
    return rule

def decision_tree_to_rules(T, idx2e, idx2nkv, request, param, empty_patterns_required):
    back, leaves = back_tree(T)
    internal_node = set()
    rules = []
    empty_patterns = []
    for n in leaves:
        e = idx2e[np.argmax(T.value[n])]
        if e and n and T.impurity[n] < param['node_impurity']:
            r = node_to_rule(n, e, back, T, request, idx2nkv, param, internal_node)
            if r:
                rules.append(r)
        else:
            while n and back[n][1] and T.impurity[ back[n][1] ] <= 0.0001:
                n = back[n][1]
            if n:
                request = Request(request)  # builds a copy 
                while n != 0: #0 is the root node
                    right, n = back[n]
                    Z = idx2nkv[T.feature[n]]
                    if isinstance(Z, tuple): #it is a feature pattern
                        m, feat, feat_value = Z
                        feat_value = feat_value.replace('"', '\\"')
                        Z = f'{m}[{feat}="{feat_value}"]'
                    if right:
                        request.append("pattern", Z)
                    else:
                        request.without(Z)  
                empty_patterns.append( request)
    if empty_patterns_required:
        return rules, empty_patterns
    return rules

def build_rules(matchings, corpus, R, param):
    features = list(feature_values_for_decision(matchings, corpus, param, ['X', 'Y']).keys())
    features_idx = e_index(features)
    X, edge_idx, y = list(), dict(), list()
    for m in matchings:
        # each matching will lead to an obs which is a list of 0/1 values
        graph = corpus[m["sent_id"]]
        nodes = m['matching']['nodes']
        obs = [0]*len(features_idx)
        for n in nodes:
            feat = graph[nodes[n]]
            for k, v in feat.items():
                if (n, k, v) in features_idx:
                    obs[features_idx[(n, k, v)]] = 1
        es = graph.edges(nodes['X'], nodes['Y'])
        if len(es) > 1:
            print("mmmmhh that should not happen")
        elif len(es) <= 1:
            e = es.pop() if es else None
            if e not in edge_idx:
                edge_idx[e] = len(edge_idx)
            y.append(edge_idx[e])
            X.append(obs)
    if not y:
        return []
    clf = DecisionTreeClassifier(max_depth=param["max_depth"],
                                max_leaf_nodes=max(y)+param["number_of_extra_leaves"],
                min_samples_leaf=param["min_samples_leaf"],
                criterion="gini")
    clf.fit(X, y)
    res = []
    idx2e = {i : v for v,i in edge_idx.items()}
    decision_tree_to_rules(clf.tree_, res, idx2e, features, R, param)
    return res
    

class Classifier():
    def __init__(self, matchings, corpus, param):
        self.fpat = list(feature_values_for_decision(matchings, corpus, param, ['X', 'Y']).keys())
        # get the list of all feature values
        self.pos = {self.fpat[i]: i for i in range(len(self.fpat))}
        X, y1, y = list(), dict(), list()
        # X: set of input values, as a list of (0,1)
        # y: set of output values, an index associated to the edge e
        # the absence of an edge has its own index
        # y1: the mapping between an edge e to some index
        for m in matchings:
            # each matching will lead to an obs which is a list of 0/1 values
            graph = corpus[m["sent_id"]]
            nodes = m['matching']['nodes']
            obs = [0]*len(self.pos)
            for n in nodes:
                feat = graph[nodes[n]]
                for k, v in feat.items():
                    if (n, k, v) in self.pos:
                        obs[self.pos[(n, k, v)]] = 1
            es = graph.edges(nodes['X'], nodes['Y'])
            if len(es) > 1:
                print("mmmmhh that should not happen")
            elif len(es) <= 1:
                e = es.pop() if es else None
                if e not in y1:
                    y1[e] = len(y1)
                y.append(y1[e])
                X.append(obs)
        if not X or not len(X[0]) or max(y) == 0:
            self.clf = None
        else:
            self.clf = DecisionTreeClassifier(max_depth=param["max_depth"],
                                              max_leaf_nodes=max(
                y)+param["number_of_extra_leaves"],
                min_samples_leaf=param["min_samples_leaf"],
                criterion="gini")
            self.clf.fit(X, y)
            self.y1 = {y1[i]: i for i in y1}

    def branches(self, pos, current, acc, threshold):
        tree = self.clf.tree_
        if tree.feature[pos] >= 0:
            if tree.impurity[pos] < threshold:
                acc[pos] = current
                return
            # there is a feature
            if tree.children_left[pos] >= 0:
                # there is a child
                left = current + ((tree.feature[pos], 1),)
                self.branches(tree.children_left[pos],
                              left, acc, threshold)
            if tree.children_right[pos] >= 0:
                right = current + ((tree.feature[pos], 0),)
                self.branches(tree.children_right[pos],
                              right, acc, threshold)
            return
        else:
            if tree.impurity[pos] < threshold:
                acc[pos] = current

    def find_classes(clf, param):
        """
    given a decision tree, extract "interesting" branches
    the output is a dict mapping the node_index to its branch
    a branch is the list of intermediate constraints = (7,1,8,0,...)
    that is feature value 7 has without clause whereas feature 8 is positive 
        """
        acc = dict()
        clf.branches(0, tuple(), acc, param["node_impurity"])
        return acc
    
    


'''