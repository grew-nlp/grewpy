import sys, os
sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew
from corpus import Corpus
import conllu

pud_file = "resources/fr_pud-ud-test.conllu"
pud = Corpus("examples/resources/fr_pud-ud-test.conllu")

print (pud.id)

print(f"nb of graph in {pud_file} = ", len(pud))

sent_id="n01027007"
graph = pud[sent_id]
print(f"nb of nodes of {sent_id} = ", len(graph))

acc = 0
for sent_id in pud:
  acc += len(pud[sent_id])
print(f"nb of nodes in {pud_file} = ", acc)

a_verb = grew.Request("X[upos=VERB]")
print(f"nb of VERB in {pud_file} = ",len(pud.search(a_verb)))

exit (0)