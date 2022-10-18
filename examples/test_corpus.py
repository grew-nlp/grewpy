import grew
import conllu

pud_file = "resources/fr_pud-ud-test.conllu"
pud = grew.Corpus("resources/fr_pud-ud-test.conllu")

print (pud.id)

print(f"nb of graph in {pud_file} = ", len(pud))

sent_id="n01027007"
graph = pud[sent_id]
print(f"nb of nodes of {sent_id} = ", len(graph))

acc = 0
for sent_id in pud:
  acc += len(pud[sent_id])
print(f"nb of nodes in {pud_file} = ", acc)

verb_pattern = grew.Pattern(("pattern",["X[upos=VERB]"]))
print(f"nb of VERB in {pud_file} = ",len(pud.search(verb_pattern)))

exit (0)


grs = grew.GRS("resources/single.grs")
#print(grs)
#grs2 = grew.GRS(grs.json())

#res = grs.run(G,"s_1")
gs = {sid : c[sid] for sid in c}
for sid in gs:
    gs[sid].sucs = {n : [] for n in gs[sid]}

c0 = {sid : gs[sid].to_conll() for sid in gs}
c1 = {sid : conllu.TokenList(c0[sid], {'text' : "None", "sent_id": sid } ) for sid in c0}
with open("empty.conllu","w",encoding="utf-8") as f:
    for sid in c1:
        f.write(c1[sid].serialize())
        f.write("\n")

ec = grew.Corpus("empty.conllu")
g0 = ec[0]
print(grs.run(g0))



"""

Plr = grew.Pattern(("pattern", "X<Y;X[upos=DET];Y[upos=NOUN]"),("without","Y-[zzz]->X"))
#Plr = grew.Pattern(("pattern", "X<Y;e:X -> Y")) #is "equivalent"
x = c.search(Plr)
print(len(x))

C = grew.Command("add_edge Y-[zzz]->X")
R = grew.Rule("R1", Plr, C)

print(G)
#G.sucs={n : [] for n in G} #clear edges
#G.sucs['4'] = [('xxx','3')]
#print(G)
Rs = grew.GRS("mygrs",rules=[R], strats=[grew.Strategy("main", "Iter(R1)")])
print("----Rs---")
print(Rs)
print("---run Rs--------")
sols = Rs.run(G,"main")
print(sols)
"""
