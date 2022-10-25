import sys,os, json

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew
import conllu

grs = grew.GRS("examples/resources/single.grs")
print (grs.index)
"""
with open("hmm.txt", "w") as f:
    f.write(json.dumps(grs.json_data(), indent=2))
"""

pud_file = "examples/resources/fr_pud-ud-test.conllu"
pud = grew.Corpus(pud_file)
sent_id = "n01027007"
graph = pud[sent_id]

Gs = grs.run(graph, strat="p_1_onf")
print(type(Gs[0]))

Plr = grew.Request(("pattern", "X<Y;X[upos=DET];Y[upos=NOUN]"), ("without", "Y-[adj]->X"))
C = grew.Command("add_edge Y-[adj]->X")
R = grew.Rule(Plr, C)
print(R)
Rs = grew.GRS(rules={'R1' : R}, strats={"main" : "Iter(R1)"})
"""
with open("hr.txt", "w") as f:
    f.write(json.dumps(Rs.json_data(), indent=2))
"""

exit(0)
