import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

import grew
from grew import Graph, Corpus, Request

pud_file = "examples/resources/pud_10.conllu"
pud = Corpus(pud_file, local=False)

def print_request_counter():
    print(f"Req: {grew.network.request_counter}")

print ("\n=============== len ===============")
print(f"nb of graph in {pud_file} = {len(pud)}")
print_request_counter()

print ("\n=============== Get one graph ===============")
sent_id="n01003013"
graph = pud[sent_id]
print(f"nb of nodes of {sent_id} = ", len(graph))
print_request_counter()

print ("\n=============== Iteration on graphs of a corpus ===============")
print ("⚠️  generate one request to Grew backend for each graph")
acc = 0
for sent_id in pud:
  acc += len(pud[sent_id])
print(f"nb of nodes in {pud_file} = ", acc)
print_request_counter()

print ("\n=============== Count request in a corpus ===============")
upos="ADV"
req = Request(f"X[upos={upos}]")

print(" ----- basic count -----")
print(f"nb of {upos} in {pud_file} = ", pud.count(req))

print (" ----- count with clustering -----")
print(f"nb of {upos} in {pud_file}, clustered by lemma:")
print (json.dumps(pud.count(req, ["X.lemma"]), indent=2))
print_request_counter()


# print ("\n=============== Search request in a corpus ===============")

# print(" ----- basic search -----")
# print(f"occurrences of {upos} in {pud_file}:")
# print (json.dumps (pud.search(req), indent=2))

# print (" ----- search with clustering -----")
# print(f"occurrences of {upos} in {pud_file}, clustered by lemma:")
# print (json.dumps (pud.search(req, ["X.lemma"]), indent=2))



def clear_edges(graph):
    for n in graph:
        graph.sucs[n] = []

req = Request(f"X -[fixed]-> Y")
print (" ----- count with clustering -----")
print(f"nb of fixed in {pud_file}, clustered by head.lemma:")
print (json.dumps(pud.count(req, ["X.lemma"]), indent=2))

g = pud[sent_id]
for n in g:
  for (s,e) in g.suc(n):
    if e  == {'1' : 'fixed'}:
      print(f"\n{n} : {g[n]} -[{e}]-> {s} : {g[s]}\n")

clear_edges(g)
pud[sent_id] = g
# NOTE: the next line does not work properly (not __set_item__ called), have a look to https://stackoverflow.com/questions/26189090/how-to-detect-if-any-element-in-a-dictionary-changes
# clear_edges(pud[sent_id]) ==> WARNING: does not change pud!

print (" ----- count with clustering -----")
print(f"nb of {upos} in {pud_file}, clustered by lemma:")
print (json.dumps(pud.count(req, ["X.lemma"]), indent=2))


exit (0)
