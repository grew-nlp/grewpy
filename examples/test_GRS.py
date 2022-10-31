import sys,os, tempfile
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
print ("  Build a GRS with explicit values")

req_det_n = grew.Request("N1[upos=DET]","N2[upos=NOUN]; N1 < N2").without("N2 -> N1")
#alt syntax: grew.Request(grew.pattern("N1[upos=DET]; N2[upos=NOUN]; N1 < N2"), grew.without("N2 -> N1"))
add_det_cde = grew.Command("add_edge N2 -[det]-> N1")
R = grew.Rule(req_det_n, add_det_cde)
grs = grew.GRS({"det":R,"s1":"det","s2":"Onf(det)","s3":"Iter(det)"})
#alt grs = grew.GRS(det=R,s1="det",s2="Onf(det)",s3="Iter(det)")
print ("------------- print (grs) -------------")
print (grs)

graph_list_1 = grs.run(graph, strat="s1")
graph_list_2 = grs.run(graph, strat="s2")
graph_list_3 = grs.run(graph, strat="s3")

print (len (graph_list_1))
print (len (graph_list_2))
print (len (graph_list_3))

print("----------------test if grs can be saved-------------")
s = str(grs)
grs2 = grew.GRS(s)
print(grs2)

print("\n============================== TEST 4 ==============================")
print("  Visiting a GRS")
print("------------- print (rules of grs) -------------")
for d in grs.rules():
    print(f"rule name: {d}\n{grs[d]} \n")

exit(0)
