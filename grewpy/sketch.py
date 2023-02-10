from .grs import Request
class Sketch:
    def __init__(self, P, cluster_criterion, avec, without, target):
        """
        sketches are defined by a pattern P, and parameters (features, edge features, etc)
        sketch name serves for rule naming
        Nodes X and Y have special status. 
        In the following, we search for 'e : X -> Y'
        Thus, in the pattern P, X, Y, e must be used with that fact in mind.
        cluster criterion is typically ["X.upos", "Y.upos"], but can be enlarged  
        """
        self.P = P
        self.cluster_criterion = cluster_criterion
        self.avec = avec
        self.without = without
        self.target = target

    def cluster(self, corpus):
        """
        search for a link X -> Y with respect to the sketch in the corpus
        we build a cluster depending on cluster criterion (e.g. X.upos, Y.upos)
        """
        P1 = self.avec(self.P)
        obs = corpus.count(P1, self.cluster_criterion, [self.target], True)
        if not obs:
            return obs
        W1 = self.without(Request(self.P))
        clus = corpus.count(W1, self.cluster_criterion, [], True)
        for L in obs:
            if L in clus:
                obs[L][''] = clus[L][tuple()]
        return obs