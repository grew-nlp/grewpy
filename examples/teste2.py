import grew

corpus = grew.Corpus("resources/fr_pud-ud-test.conllu")
graph = corpus["n01027007"]


print ("=== test 1 ===")
single = grew.GRS("resources/single.grs")

try:
    res = single.run(graph)
except:
    print ("Erreur: strat non définie ?")

print ("=== test 2 ===")

res = single.run (graph, "s_1")
print (res)

print ("=== test 3 ===")

grs =  grew.GRS("strat main {Seq () }")
res = grs.run (graph)
print (res)
