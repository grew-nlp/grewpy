import sys,os
sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew


corpus = grew.Corpus("examples/resources/test1.conllu")
graph = corpus[0]
print (len(graph))

print ("\n============================== TEST 1 ==============================")
print ("  Build a GRS from a file (examples/resources/test1.grs)")
grs = grew.GRS("examples/resources/test1.grs")
print ("------------- print (grs) -------------")
print (grs)


graph_list_1 = grs.run(graph, strat="s1")
graph_list_2 = grs.run(graph, strat="s2")
graph_list_3 = grs.run(graph, strat="s3")

print (len (graph_list_1))
print (len (graph_list_2))
print (len (graph_list_3))


print ("\n============================== TEST 2 ==============================")
print ("  Build a GRS from a string")
string_grs = """
rule det {
  pattern { N1[upos=DET]; N2[upos=NOUN]; N1 < N2 }
  without { N2 -> N1 }
  commands { add_edge N2 -[det]-> N1}
}

strat s1 { det }
strat s2 { Onf (det) }
strat s3 { Iter (det) }
"""

grs = grew.GRS(string_grs)
print ("------------- print (grs) -------------")
print (grs)


graph_list_1 = grs.run(graph, strat="s1")
graph_list_2 = grs.run(graph, strat="s2")
graph_list_3 = grs.run(graph, strat="s3")

print (len (graph_list_1))
print (len (graph_list_2))
print (len (graph_list_3))


print ("\n============================== TEST 3 ==============================")
print ("  Build a GRS from a JSON")



json_grs = {
    "decls": {
        "det": {
            "request": [
                { "pattern": ["N1[upos=DET]", "N2[upos=NOUN]", "N1 < N2"] },
                { "without": ["N2 -> N1"] }
            ],
            "commands": [
                "add_edge N2 -[det]-> N1"
            ]
        },
        "s1": "det",
        "s2": "Onf (det)",
        "s3": "Iter (det)",
    }
}

grs = grew.GRS(json_grs)
print ("------------- print (grs) -------------")
print (grs)


graph_list_1 = grs.run(graph, strat="s1")
graph_list_2 = grs.run(graph, strat="s2")
graph_list_3 = grs.run(graph, strat="s3")

print (len (graph_list_1))
print (len (graph_list_2))
print (len (graph_list_3))
exit (0)







pud_file = "examples/resources/fr_pud-ud-test.conllu"
pud = grew.Corpus(pud_file)
sent_id = "n01027007"
graph = pud[sent_id]

print ("\n ## 2 ------------- after run -------------")
Gs = grs.run(graph, strat="p_1_onf")
print(type(Gs[0]))

Plr = grew.Request(("pattern", "X<Y;X[upos=DET];Y[upos=NOUN]"), ("without", "Y-[adj]->X"))
C = grew.Command("add_edge Y-[adj]->X")
R = grew.Rule(Plr, C)
r2 = grew.Rule(Plr, grew.Command("add_edge X-[adj]-Y;del_edge X-[obl]->Y"))
#print(R)
Rs = grew.GRS(rules={'R1' : R}, strats={"main" : "Iter(R1)"})

print ("\n ## 3 ------------- print locally built RS -------------")
print(Rs)


print ("\n ## 4 ------------- print items in Rs -------------")
for d in Rs:
    print(f"{d} {Rs[d]}")

print ("\n ## 5 ------------- print rules in Rs -------------")
for d in Rs.rules():
    print(d)

grs["R2"] = r2
print ("\n ## 6 ------------- print modified grs -------------")
print(grs)
print (grs.index)
"""
with open("hr.txt", "w") as f:
    f.write(json.dumps(Rs.json_data(), indent=2))
"""

exit(0)
