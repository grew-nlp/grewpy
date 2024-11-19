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
        def _none_(n):
            return '' if '__none__' in n else n[0]
        obs2 = corpus.count(self.P, self.cluster_criterion, ["X->Y"], True)
        obs3 = dict()
        for k1, o1 in obs2.obs.items():
            if len(o1) == 1 and ('__none__',) in o1:
                ...
            else:
                obs3[k1] = {_none_(k2) : v for k2,v in o1.items()}
        return obs3