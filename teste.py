import grew

c = grew.Corpus("UD_French-PUD/fr_pud-ud-test.conllu")
print(len(c))
for sid in c:
    print(len(c[sid]),end=" ")
x = c[33]
with open("hum.dot","w") as f:
    f.write(x.to_dot())

for n in c[-1]:
    print(f"node={n}")

p = grew.Pattern("pattern {X[upos=VERB]}")
print(len(c.search(p)))

grs = grew.GRS("single.grs")
print(grs.json())

