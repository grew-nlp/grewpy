import sys, os

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew
from grew import Corpus, Request

r = Request("X<Y")
c = Corpus("examples/resources/fr_pud-ud-test.conllu")

print(c.count(r))
