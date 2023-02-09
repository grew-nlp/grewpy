import sys,os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))  # Use local grew lib

from grewpy import Corpus, GRSDraft, Rule, Request, Commands, GRS, set_config
from grewpy import Grew_web
set_config("sud")

corpus = Corpus("examples/resources/pud_10.conllu")

grs_text = """
rule r { pattern { N[upos="NOUN"] } commands { N.upos=N } }
strat main { Onf (r) }
"""
grs = GRS(grs_text)

web = Grew_web()
web.load_corpus(Corpus(corpus[0:3])) #to get a few samples in the corpus
web.load_grs(grs)

print(web.url())

# TEST 1: start Grew-web with a GRS and a corpus
#grs.grew_web(corpus)

# TEST 2: start Grew-web with a GRS
# grs.grew_web()

# TEST 3: start Grew-web with a corpus
# corpus.grew_web()

