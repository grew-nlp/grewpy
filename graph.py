"""
Grew module: anything you want to talk about graphs
Graphs are represented either by a dict (called dict-graph),
or by an str (str-graph).
"""
import os.path
import re
import copy
import tempfile
import json

#from grew import utils
#from grew import network
import utils
import network

''' interfaces'''

class Graph(dict):
    """
    a dict mapping node keys to feature structure
    with an extra dict mapping node keys to successors (pair of edge feature,node key)
    """
    def __init__(self,data=None):
        """
        :param data: either None=>empty graph
        a string: a json representation => read json
        a json-decoded representation => fill with json
        an oterh graph => copy the dict 
        :return: a graph
        """
        super().__init__()
        self.edges = dict()
        if data is None:
            pass            
        elif isinstance(data,str):
            #either json or filename
            try:
                data = json.loads(data)
                for name, val in data.items():
                    self[name] = val[0]
                    self.edges[name] = val[1]
            except json.decoder.JSONDecodeError:
                pass
        elif isinstance(data, Graph):
            super().__init__(data)
            self.edges = copy(data.edges)
        elif isinstance(data, dict):
            #supposed to be json decoded str
            for name,val in data.items():
                self[name] = val[0]
                self.edges[name] = val[1]
            for n,v in self.edges.items():
                for e in v:
                    assert len(e) == 2

    def to_dot(self):
        """
        return a string in dot/graphviz format
        """
        s = 'digraph G{\n'
        for n,fs in self.items():
            s += f'{n}[label="'
            label = ["%s:%s" % (f,v.replace('"','\\"')) for f, v in fs.items()]
            s += ",".join(label)
            s += '"];\n'
        s += "\n".join([f'{n} -> {m}[label="{e}"];' for n,suc in self.edges.items() for e,m in suc])
        return s + '\n}'



def save(gr, filename):
    req = { "command": "save_graph", "graph": json.dumps(gr), "filename": filename }
    reply = network.send_and_receive(req)
    return

def add_node(g, s, a):
    """
    Add a node s labeled a in graph g
    """
    g[s] = (a, []) # ERROR

def add_edge(gr, source, label, target):
    """
    Add an edge from [source] to [target] labeled [label] within [gr]
    :param gr: the graph
    :param source: the source node id
    :param label: the label of edge between [source] and [target]
    :param target: the target node id
    :return:
    """
    if source not in gr:
        raise utils.GrewError({"function": "grew.add_edge", "src": source, "message":"KeyError"})
    elif target not in gr:
        raise utils.GrewError({"function": "grew.add_edge", "tar": target, "message":"KeyError"})
    else:
        succs = gr[source][1]
        if not (label, target) in succs:
            succs.append((label, target))


def insert_before(gr, label, pivot):
    """
    Add a new node to the ordered graph
    :param gr: an ordered graph
    :param label: the label of the new node
    :param pivot: the new node will be put just before the node [pivot]
    :return: the id of the new node
    """
    leftid = glb(gr, pivot)
    nid = mid(leftid, pivot) if leftid else left(pivot)
    gr[nid] = ('label="%s"'%(label), [])
    return nid

def insert_after(gr, label, pivot):
    """
    Add a new node to the ordered graph
    :param gr: an ordered graph
    :param label: the label of the new node
    :param pivot: the new node will be put just after the node [pivot]
    :return: the id of the new node
    """
    rightid = lub(gr, pivot)
    nid = mid(rightid, pivot) if rightid else right(pivot)
    gr[nid] = ('label="%s"'%(label), [])
    return nid

_canvas = None
def draw(gr,format="dep"):
    global _canvas
    """Opens a window with the graph."""

    if format == "dep":
        png_file = dep_to_png(gr)
    elif format == "dot":
        png_file = dot_to_png(gr)
    else:
        raise GrewError('Unknown format: %s' % format)

    if not _canvas:
        import tkinter
        from tkinter import Tk, Canvas, PhotoImage, NW
        app = Tk()
        _canvas = Canvas(app, width=900, height=500)
        _canvas.pack()
        pic = PhotoImage(file=png_file)
        _canvas.create_image(0, 0, anchor=NW, image=pic)
        app.mainloop()

def dot_to_png(gr):
    req = { "command": "dot_to_png", "graph": json.dumps(gr) }
    return network.send_and_receive(req)

def dep_to_png(gr):
    req = { "command": "dep_to_png", "graph": json.dumps(gr) }
    return network.send_and_receive(req)


def search(pattern, gr):
    """
    Search for [pattern] into [gr]
    :param patten: a string pattern
    :param gr: the graph
    :return: the list of matching of [pattern] into [gr]
    """
    try:
        req = {
            "command": "search",
            "graph": json.dumps(gr),
            "pattern": pattern
        }
        reply = network.send_and_receive(req)
        return reply
    except utils.GrewError as e:
        raise utils.GrewError({"function": "grew.search", "message": e.value})


def graph_svg(graph):
    req = {
        "command": "dep_to_svg",
        "graph": json.dumps(graph),
    }
    return network.send_and_receive(req)


def graph2dot(graph):
    """
    Transformation of a graph to a graphviz string
    :param graph: a graph (dict)
    :return: string
    """

