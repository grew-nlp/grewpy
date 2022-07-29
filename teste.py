import grew

grew.init()
c = grew.Corpus("UD_French-PUD/fr_pud-ud-test.conllu")
print(len(c))
print(c[33])
for sid in c:
    print(len(c[sid]),end=" ")

for n in c[-1]:
    print(f"node={n}")

p = grew.Pattern("pattern {X[upos=VERB]}")
print(len(c.search(p)))

grs = grew.GRS("single.grs")
print(grs.json())