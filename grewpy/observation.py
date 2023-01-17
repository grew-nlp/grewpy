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

    def __init__(self, obs, parameter, keys):
        if parameter:
            intermediate = Observation.flatten(obs, parameter)
            if keys:
                self.obs = {L : Observation.flatten(V, keys) for L,V in intermediate.items()}
            else:
                self.obs = intermediate
        else:
            if keys:
                self.obs = Observation.flatten(obs, keys)
            else:
                self.obs = obs

    def __iter__(self):
        return iter(self.obs)

    def __getitem__(self, k):
        return self.obs[k]

    def __bool__(self):
        return bool(self.obs)

    def anomaly(self, L,  threshold):
        """
        L is a key within self
        return for L an edge and its associated probability if beyond base_threshold
        """
        s = sum(self.obs[L].values())
        for x, v in self.obs[L].items():
            if v > threshold * s and x:
                return x, v/s
        return None, None
