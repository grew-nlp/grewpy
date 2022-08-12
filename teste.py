import grew

c = grew.Corpus("UD_French-PUD/fr_pud-ud-test.conllu")
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

grs = grew.GRS("single.grs")
print(grs)
#grs2 = grew.GRS(grs.json())

#print(grs2.index)
G = c["n01027007"]
print(len(G))
#res = grs.run(G,"s_1")

Plr = grew.Pattern(("pattern", "X<Y"), ("pattern", "e:X -> Y"))
#Plr = grew.Pattern(("pattern", "X<Y;e:X -> Y")) #is "equivalent"
print(len(c.search(Plr)))
print(Plr)

C = grew.Command("add_edge X-[hum]->Y")
R = grew.Rule("R1", Plr, C)

Rs = grew.GRS("mygrs",rules=[R], strats=[grew.Strategy("main", "Onf(R1)")])
print("----Rs---")
print(Rs)
print("---run Rs--------")
print(Rs.run(G,"main"))
