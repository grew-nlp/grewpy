from sklearn.tree import export_graphviz
import argparse
import logging

from rule_forgery import observations, clf_dependency

# Use local grew lib
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from grewpy import Request, CorpusDraft
from grewpy import Corpus, set_config
from grewpy.graph import Fs_edge

def clear_but_working(g):
    """
    delete edges within g
    """
    g.sucs = {n : [] for n in g.sucs}
    return (g)

def basic_edges(g):
    """
    change edges {1:comp, 2:obl} to {1:comp}
    """
    g.sucs = {n : [(m,  Fs_edge({'1': e['1']}) ) for m, e in g.sucs[n] ] for n in g.sucs}
    return g


def check(corpus, sketch, excluded, edge):
    """
    return true if no pattern within excluded has an edge 
    """
    found = False
    for request in excluded:
        if edge:
            req = Request(request).without(f'X-["{edge}"]->Y')
        else:
            req = Request(request).with_(f'X->Y')
        lines = corpus.search(req)
        if lines:
            found = True
            print(f"forbidden pattern\n{request}\n")
            print(f"""sent_id : {",".join(f"{match['sent_id']}" for match in lines)}\n""")
    return not found

def learn_zero_knowledge(gold, args, sketch, param):
    nodes = sketch.named_entities()['nodes']
    _, X, y, edge_idx, nkv_idx = observations(gold, sketch, nodes, param)
    idx2nkv = {v:k for k,v in nkv_idx.items()}
    dep = Fs_edge(args.dependency) if args.dependency else None
    if dep not in edge_idx:
        print(f"no such dependency in corpus (e.g. subj): {args.dependency}")
        sys.exit(2)
    requests, clf = clf_dependency(edge_idx[dep], X, y, idx2nkv,sketch,nodes,param,args.depth, args.branch_details)
    if args.export_tree:
        export_graphviz(clf, out_file=args.export_tree, 
                        feature_names=[str(idx2nkv[i]) for i in range(len(idx2nkv))])    
    if args.file:
        with open(args.file, "w") as f:
            for pattern in requests:
                f.write(f"%%%\n{str(pattern)}\n")
    print("testing patterns")            
    if check(corpus_gold, sketch, requests, args.dependency):
        print("self verification: no forbidden patterns found")

def parse_request(filename):
    """
    parsing an (anti-)pattern file 
    """
    def body(line): return line[line.index('{')+1:].strip().strip('}')
    excluded = []
    current_pattern = None
    with open (filename) as f: #we parse the file
        for line in f:
            if '%%%' in line:
                if current_pattern:
                    excluded.append(current_pattern)
                current_pattern = Request()
            else:
                if "pattern" in line:
                    current_pattern.append("pattern", body(line))
                elif "without" in line:
                    current_pattern.without(body(line))
                else:
                    logging.critical(f"oups {line} is not in a pattern")
    if current_pattern:
        excluded.append(current_pattern)
    return excluded


def verify(corpus, args):
    """
    for requests within args.file, verifies whether we found them
    in the corpus
    """
    excluded = parse_request(args.file)
    print("testing patterns")
    if check(corpus,excluded):
        print("nothing found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='antipattern.py',
                                     description='Learn/Test antipattern on a corpus')
    parser.add_argument('action', help="action in [learn, verify]")
    parser.add_argument('corpus', help='a conll file')
    parser.add_argument('-f', '--file', default=None, help="filename for the patterns")
    parser.add_argument('-t', '--threshold', default=1e-10, type=float, help="minimal threshold to consider a node as pure")
    parser.add_argument('-s','--nodetails',action="store_true", help="simplifies edges: replace comp:obl by comp")
    parser.add_argument('-d', '--depth', default=8, help="depth of the binary decision tree")
    parser.add_argument('-b', '--branch_details', action="store_true", help="if set, requests are fully decomposed")  
    parser.add_argument('--request',default='', help='request file') 
    parser.add_argument('--export_tree', default='', help='export classifier tree as a dot file')
    parser.add_argument('--dependency', default='', help='export classifier tree as a dot file')
    args = parser.parse_args()
    set_config("sud")
    param = {
        "min_samples_leaf": 5,
        "skipped_features":  {'xpos', 'SpaceAfter', 'Shared', 'textform', 'Typo', 'form', 'wordform', 'CorrectForm'},
        "node_impurity": 0.15,
        "threshold" : args.threshold,
        "tree_depth" : args.depth,
        "ratio": 50
    }
    corpus_gold = Corpus(args.corpus)
    if args.nodetails:
        corpus_gold = Corpus(CorpusDraft(corpus_gold).apply(basic_edges))
    if args.action == 'learn':
        if args.request:
            with open(args.request) as f:
                request = Request(f.read())
        else:
            request = Request('pattern { X[];Y[] }')
        learn_zero_knowledge(corpus_gold, args, request, param)
    elif args.action == 'verify':
        verify(corpus_gold, args)
    else:
        logging.critical('action {args.action} is not known. Use either learn or verify.')
