import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

from grew import Corpus, Request

import conllu

pud_file = "examples/resources/fr_pud-ud-test.conllu"
pud = Corpus(pud_file)

#print (f"pud.id = {pud.id}")

print ("\n=============== len ===============")
print(f"nb of graph in {pud_file} = {len(pud)}")

print ("\n=============== Get one graph ===============")
sent_id="n01027007"
graph = pud[sent_id]
print(f"nb of nodes of {sent_id} = ", len(graph))

print ("\n=============== Iteration on graphs of a corpus ===============")
print ("⚠️  generate one request to Grew backend for each graph")
acc = 0
for sent_id in pud:
  acc += len(pud[sent_id])
print(f"nb of nodes in {pud_file} = ", acc)

print ("\n=============== Count request in a corpus ===============")
upos="SYM"
req = Request(f"X[upos={upos}]")

print(" ----- basic count -----")
print(f"nb of {upos} in {pud_file} = ", pud.count(req))

print (" ----- count with clustering -----")
print(f"nb of {upos} in {pud_file}, clustered by lemma:")
print (json.dumps(pud.count(req, ["X.lemma"]), indent=2))



print ("\n=============== Search request in a corpus ===============")

print(" ----- basic search -----")
print(f"occurrences of {upos} in {pud_file}:")
print (json.dumps (pud.search(req), indent=2))

print (" ----- search with clustering -----")
print(f"occurrences of {upos} in {pud_file}, clustered by lemma:")
print (json.dumps (pud.search(req, ["X.lemma"]), indent=2))


exit (0)
