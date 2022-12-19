import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

from grewpy import Graph, CorpusDraft, Request, Corpus, request_counter

pud_file = "examples/resources/pud_10.conllu"
pud = Corpus(pud_file)

print ("\n=============== len ===============")
print(f"nb of graph in {pud_file} = {len(pud)}")
print (request_counter())

print ("\n=============== Get one graph ===============")
sent_id="n01003013"
graph = pud[sent_id]
print(f"nb of nodes of {sent_id} = ", len(graph))
print (request_counter())

print(f"len(pud[0]) = {len(pud[0])}")
print(f"len(pud[-1]) = {len(pud[-1])}")
print(f"[len(g) for g in pud[-3:]] = {[len(g) for g in pud[-3:]]}")
#other forms pud[-3:-1], pud[1:7:2], ...


print ("\n=============== Iteration on graphs of a corpus ===============")
print ("⚠️  generate one request to Grew backend for each graph")
acc = 0
for sent_id in pud.get_sent_ids():
  acc += len(pud[sent_id])
print(f"nb of nodes in {pud_file} = ", acc)
print (request_counter())

print ("\n=============== Count request in a corpus ===============")
upos="ADV"
req = Request(f"X[upos={upos}]")

print(" ----- basic count -----")
print(f"nb of {upos} in {pud_file} = ", pud.count(req))

print (" ----- count with clustering -----")
print(f"nb of {upos} in {pud_file}, clustered by lemma:")
print (json.dumps(pud.count(req, ["X.lemma"]), indent=2))
print (request_counter())


corpus = CorpusDraft(pud)
print("\n=============== Iteration on graphs of a corpus ===============")
acc = 0
for sent_id in corpus:
  acc += len(corpus[sent_id])
print(f"nb of nodes in {pud_file} = ", acc)
print (request_counter())

def clear_edges(graph):
    for n in graph:
        graph.sucs[n] = []

for sid in corpus:
  clear_edges(corpus[sid])

noedge_corpus = Corpus(corpus)
print(" ----- counting nsubj within corpus -----")
dep = "nsubj"
req = Request(f"X[];Y[];X -[{dep}]-> Y")
print(f"nb of {dep} in {pud_file} = ", pud.count(req))
print(f"nb of {dep} in noedge_corpus = ", noedge_corpus.count(req))

exit(0)



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
#pud[sent_id] = g
# NOTE: the next line does not work properly (not __set_item__ called), have a look to https://stackoverflow.com/questions/26189090/how-to-detect-if-any-element-in-a-dictionary-changes
# clear_edges(pud[sent_id]) ==> WARNING: does not change pud!

