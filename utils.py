import re
import math
import json

'''
Identifier tools
'''

def id2float(name):
    m = re.search(r'[a-zA-Z]*(\d+\.?\d*)', name)
    if m:
        return float(m.group(1))
    else:
        print(r"[GREW] Node identifier '%s' does not follow grammar [a-zA-Z]*(\d+\.?\d*)" % name)


def float2id(f):
    a = int(f) if f.is_integer() else f
    return 'W%s' % a


def id_lt(n1, n2):
    return id2float(n1) < id2float(n2)


def mid(n1, n2):
    return float2id((id2float(n1) + id2float(n2)/2))


def left(n1):
    f1 = id2float(n1)
    if f1.is_integer():
        return float2id(f1-1)
    else:
        return float2id(math.floor(f1))


def right(n1):
    f1 = id2float(n1)
    if f1.is_integer():
        return float2id(f1+1)
    else:
        return float2id(math.ceil(f1))



''' Graph utility tools '''

class GrewError(Exception):
    """A wrapper for grew-related errors"""

    def __init__(self, message):
        self.value = message
    def __str__(self):
        if isinstance(self.value, dict):
            return ("\n".join (("", "-"*80, json.dumps(self.value, indent=2), "-"*80)))
        else:
            return ("\n".join (("", "="*80, str (self.value), "="*80)))

GrewError.__doc__ = "A wrapper for grew-related errors"


def glb(gr, pivot):
    """
    Greatest lower bound: the greatest node lower than [pivot] in [gr]
    :param gr: the dict-graph
    :param pivot: the pivot node
    :return: the greatest lower bound if it exists, otherwise None
    """
    res = None
    for nid in gr:
        if res and id_lt(res, nid) and id_lt(nid, pivot):
            res = nid
        elif id_lt(nid, pivot):
            res = nid
    return res


def lub(gr, pivot):
    """
    Least upper bound: the smallest node greater than [pivot] in [gr]
    :param gr: the dict-graph
    :param pivot: the pivot node
    :return: the least upper bound if it exists, otherwise None
    """
    res = None
    for nid in gr:
        if res and id_lt(nid, res) and id_lt(pivot, nid):
            res = nid
        elif id_lt(pivot, nid):
            res = nid
    return res


def rm_dups(t):
    s = []
    for i in t:
        if i not in s:
            s.append(i)
    return s
