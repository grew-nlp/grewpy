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
    s = 'graph G{\n'
    for n in graph:
        s += '%s [label="%s"]\n' % (n, graph[n][0])
    for n in graph:
        for (nid, lab) in graph[n][1]:
            s += '%s->%s[label="%s"]\n' % (n, nid, lab)
    return s + '}'

def graph(data=None):
    """
    :param data: either a list of string,
                 or a string in GREW format
                 or a filename in GREW format
                 or another graph
    :return: a graph
   """
    try:
        if not data:
            return dict()
        if isinstance(data, list):
            # builds a flat ordered (using list order) graph
            return {float2id(float(i)): (data[i], []) for i in range(len(data))}
        elif isinstance(data, str):
            # build from a JSON string
            try:
                return json.loads(data)
            except json.decoder.JSONDecodeError:
                if os.path.isfile(data):
                    req = { "command": "load_graph", "filename": data }
                    reply = network.send_and_receive(req)
                else:
                    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".gr") as f:
                        f.write(data)
                        f.seek(0)  # to be read by others
                        req = { "command": "load_graph", "filename": f.name }
                        reply = network.send_and_receive(req)
                return (reply)
    
        elif isinstance(data,dict):
            # copy an existing graph
            return copy.deepcopy(data)
        else:
            raise GrewError('Library call error')
    except utils.GrewError as e: 
        raise utils.GrewError({"function": "grew.graph", "data": data, "message":e.value})

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
