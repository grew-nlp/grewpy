import grew
import conllu

c = grew.Corpus("test.conllu") #UD_French-PUD/fr_pud-ud-test.conllu")
# print(len(c))
# for sid in c:
#     print(len(c[sid]),end=" ")
# x = c[33]
"""
with open("hum.dot","w") as f:
    f.write(x.to_dot())

for n in c[-1]:
    print(f"node={n}")
"""

# p = grew.Pattern(("pattern",["X[upos=VERB]"]))
# print(len(c.search(p)))

#grs = grew.GRS("r0.grs")
#print(grs)
#grs2 = grew.GRS(grs.json())

#print(grs2.index)
G = c["n01001011"]
print(len(G))
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
#print(grs.run(g0))



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
