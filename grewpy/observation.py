
class Observation:
    """
    maps a tuple of criteria to a dict mapping edge -> nb of observation
    """
    @staticmethod
    def flatten(obs, crit):
        def _flatten(obs, crit, L): #flatten a dict according to crit
            if not obs:
                return
            if not crit:
                yield (L, obs)
            else:
                for k in obs:
                    yield from _flatten(obs[k], crit[1:], L + (k,))
        return {L : v for L, v in _flatten(obs, crit, tuple())}

    def __init__(self, **kwargs):
        if "obs" in kwargs:
            if "parameter" in kwargs:
                intermediate = Observation.flatten(kwargs["obs"], kwargs["parameter"])
            if "keys" in kwargs:
                self.obs = {L : Observation.flatten(V, kwargs["keys"]) for L,V in intermediate.items()}
            else:
                self.obs = intermediate
        else:
            self.obs = dict()


    def __ior__(self, other):
        for parameter_keys in other.obs:
            if parameter_keys not in self.obs:
                self.obs[parameter_keys] = dict()
            for observation_keys in other.obs[parameter_keys]:
                if observation_keys not in self.obs[parameter_keys]:
                    self.obs[parameter_keys][observation_keys] = 0
                self.obs[parameter_keys][observation_keys] += other.obs[parameter_keys][observation_keys]
        return self

    def __iter__(self):
        return iter(self.obs)

    def __getitem__(self, k):
        return self.obs[k]

    def __setitem__(self, k, v):
        self.obs.__setitem__(k,v)

    def __bool__(self):
        return bool(self.obs)

    def anomaly(self, L,  threshold):
        """
        L is a key within self
        return for L an edge and its occurrence evaluation 
        and number of total occurrences if beyond base_threshold
        """
        s = sum(self.obs[L].values())
        for x, v in self.obs[L].items():
            if v > threshold * s and x:
                return (x,v,s)
        return None, None, None

  
    def zipf(observation, n, k, width, ratio):
        """
        return the list of width best features
        if they are beyond ratio
        """
        if len(observation[(n, k)]) < 1:
            return []  # no values or 1 is not sufficient
        values = list(observation[(n, k)].keys())
        values.sort(reverse=True)
        occs = sum([observation[(n, k)][v] for v in values])
        zoccs = sum([observation[(n, k)][v] for v in values[0:width]])
        if zoccs/occs > ratio:
            return values[:width]
        return []
