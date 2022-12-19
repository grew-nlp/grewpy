import sys
import json
txt = open(sys.argv[1]).read()
graph = json.loads(txt)

f =open(sys.argv[2], "w")
f.write("Digraph G{\n")
f.write("rankdir=LR;\n")
f.write("node[shape=box];\n")

for k in graph:
  for s in graph[k]:
    if "builtin" not in s:
      f.write(f'"{k}" -> "{s}";\n')

f.write("}")
f.close()