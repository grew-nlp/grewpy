import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

import grewpy
from grewpy import Corpus, CorpusDraft, Request, grew_web
from grewpy.grew_web import Grew_web

def add_span(g):
    left, right = {i: i for i in g}, {i: i for i in g}
    todo = [i for i in g]
    g._sucs = {i : g._sucs.get(i,[]) for i in g}
    while todo:
        n = todo.pop(0)
        for s, _ in g.sucs[n]:
            if g.lower(left[s], left[n]):
                left[n] = left[s]
                todo.append(n)
            if g.greater(right[s], right[n]):
                right[n] = right[s]
                todo.append(n)
    for i in g:
        g.sucs[i] += [(left[i], {'1': 'LEFT_SPAN'}),
                      (right[i], {'1': 'RIGHT_SPAN'})]
    return g


#corpus = Corpus("resources/fr_pud-sud-test.conllu")
corpus = Corpus("de_gsd-sud-test.conllu")
#corpus = Corpus(c.apply(add_span))  # span

req_root_verb = Request("V[upos=VERB|AUX];R-[root]->V")
res = corpus.search(req_root_verb)

corpus_filtered = Corpus({item["sent_id"] : corpus[item["sent_id"]] for item in res})
print(len(corpus_filtered))
pos2 = Request("R-[root]->V;V[upos=VERB|AUX];V-[^punct|discourse|cc]->L;L<<V")\
    .without("V-[^punct|discourse|cc]->L2;L2<<V").without(
    "V->T;T->L;V<<T;L<<V"
    )
pos3 = Request("R-[root]->V;V[upos=VERB|AUX];V-[^punct|discourse|cc]->L1;L1<<V;V-[^punct|discourse|cc]->L2;L2<<L1")\
    .without("V-[^punct|discourse|cc]->L3;L2<<L3;L3<<V")
pos1 = Request("R-[root]->V;V[upos=VERB|AUX]").without("V-[^punct|discourse|cc]->L;L<<V").without(
    "V[Mood=Imp]")
ppos1 = Request("R-[root]->V;V[upos=VERB|AUX]").without(
    "V-[^punct|discourse|cc]->L;L<<V").without(
    "V -> T;T -> L;V << T;L << V")
gauche0 = corpus_filtered.count(pos1)
ggauche0 = corpus_filtered.count(ppos1)
gauche1 = corpus_filtered.count(pos2)
gauche2p = corpus_filtered.search(pos3)

print(f"zero = {gauche0}, zero = {ggauche0},1 = {gauche1}, 2 = {len(gauche2p)}")
print(gauche2p[0:10])


web = Grew_web()
print(web.url())
web.load_corpus(corpus)
