import sys
import os.path

sys.path.insert(0,os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

from grew import Graph

g_str = """
{
  "nodes": {
    "A": "A",
    "B": "B",
    "C": "C"
  },
  "edges": [
    { "src": "A", "label": "X", "tar": "B"},
    { "src": "B", "label": "Y", "tar": "C"},
    { "src": "C", "label": "Z", "tar": "A"}
  ],
  "order": [ "A", "B" ]
}
"""

print ("---- build a graph from a string ----")
g1 = Graph(g_str)

print (f"|nodes| = {(len (g1))}")

print (g1["A"])
print ("---- sucs 2 ----")
print (g1.suc("A"))
print ("---- sucs 1 ----")
print (g1.suc("B"))


print ("---- JSON output of a graph ----")
print (g1.json())
