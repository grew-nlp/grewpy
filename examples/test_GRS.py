import sys,os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))  # Use local grew lib

from grewpy import Corpus, GRSDraft, Rule, Request, Commands, GRS, set_config

set_config("sud")

corpus = Corpus("examples/resources/test1.conllu")
graph = corpus[0]

print ("\n============================== TEST 1 ==============================")
print ("  Build a GRS from a file (examples/resources/test1.grs)")
grs = GRS("examples/resources/test1.grs")
print ("------------- print (grs) -------------")
print (grs)

print ("nb of output with strat s1 (should be 2) ---> ", end='')
print (len (grs.run(graph, strat="s1")))
print ("nb of output with strat s2 (should be 1) ---> ", end='')
print (len (grs.run(graph, strat="s2")))
print ("nb of output with strat s3 (should be 1) ---> ", end='')
print (len (grs.run(graph, strat="s3")))

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

grs = GRS(string_grs)
print ("------------- print (grs) -------------")
print (grs)

print ("nb of output with strat s1 (should be 2) ---> ", end='')
print (len (grs.run(graph, strat="s1")))
print ("nb of output with strat s2 (should be 1) ---> ", end='')
print (len (grs.run(graph, strat="s2")))
print ("nb of output with strat s3 (should be 1) ---> ", end='')
print (len (grs.run(graph, strat="s3")))

print ("\n============================== TEST 3 ==============================")
print ("  Build a GRS with explicit values")

req_det_n = Request("N1[upos=DET]","N2[upos=NOUN]; N1 < N2").without("N2 -> N1")
add_det_cde = Commands("add_edge N2 -[det]-> N1")
R = Rule(req_det_n, add_det_cde)
grs = GRSDraft({"det":R,"s1":"det","s2":"Onf(det)","s3":"Iter(det)"})
print ("------------- print (grs) -------------")
print (grs)
print("-------- print corresponding abstract grs--------")
agrs = GRS(grs)
print(agrs)

print("----------- run grs with some strategies-----------")
print ("nb of output with strat s1 (should be 2) ---> ", end='')
print (len (agrs.run(graph, strat="s1")))
print ("nb of output with strat s2 (should be 1) ---> ", end='')
print (len (agrs.run(graph, strat="s2")))
print ("nb of output with strat s3 (should be 1) ---> ", end='')
print (len (agrs.run(graph, strat="s3")))

print("----------------test if grs can be saved-------------")
s = str(grs)
grs2 = GRSDraft(s)
print(grs2)

print("\n============================== TEST 4 ==============================")
print("  Visiting a GRS")
print("------------- print (rules of grs) -------------")
for d in grs.rules():
    print(f"rule name: {d}\n{grs[d]} \n")

exit(0)
