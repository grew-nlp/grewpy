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

def feature_values_for_decision(matchings, corpus, param, nodes):
    """
    restrict feature values to those useful for a decision tree
    """
    observation = feature_value_occurences(matchings, corpus)
    features = dict()
    for (n,k) in observation:
        if n in nodes and k not in param["skip_features"]:
            for v in observation.zipf(n, k, param["feat_value_size_limit"], param["zipf_feature_criterion"]):
                features[(n,k,v)] = observation[(n,k)][v]
    return features

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
            es = {e for e in graph.edges(
                nodes['X'], nodes['Y']) if "rank" in e}
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
