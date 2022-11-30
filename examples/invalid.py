import sys, os

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew
from grew import CorpusDraft, Request

r = Request("X[]; Y[]; X < Y")
c = CorpusDraft("examples/resources/fr_pud-ud-test.conllu")

print(c.count(r))
