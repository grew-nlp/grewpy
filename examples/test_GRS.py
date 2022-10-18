import grew
import conllu

grs = grew.GRS("resources/single.grs")
print(type(grs))
print (grs.index)
#grs2 = grew.GRS(grs.json())

exit (0)

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
