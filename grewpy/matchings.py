from .observation import Observation

class Matching():
    """
    a class that represent a matching
    """
    def __init__(self, json_data, graph):
        self.nodes = json_data["nodes"]
        self.edges = json_data["edges"]
        self.graph = graph

    def feature_values(self, arg=None, flat=False):
        """
        return the list of feature values of arg
        arg = None = all nodes, an explicit list of nodes, a unique node
        if flat, all values are flatten, otherwise, computation is done for each node
        """
        nl = self.nodes if not arg else arg if isinstance(arg, list) else [arg]
        if flat:
            """
            not yet implemented
            """
            ...
        else:
            observation = Observation()
            for n in nl:
                n_in_g = self.nodes[n]
                for k,v in self.graph[n_in_g].items():
                    if (n, k) not in observation:
                        observation[(n, k)] = dict()
                    observation[(n, k)][v] = observation[(n, k)].get(v, 0)+1
            return observation


class Matchings(dict):
    """
    matching sentence ids to list of matchings
    """
    def __init__(self, json_data, corpus):
        super().__init__()
        for line in json_data:
            sid = line["sent_id"]
            if sid not in self:
                self[sid] = []
            self[sid].append(Matching( line["matching"], 
            corpus[sid]))

    def feature_values(self, arg=None, flat=False):
        observation = Observation()
        for ms in self.values():
            for m in ms: #for each matching of the sentence id
                observation |= m.feature_values(arg,flat)
        return observation

    

