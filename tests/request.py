import unittest
import sys, os
import wget
import os.path

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib

from grewpy import Request, Corpus, set_config
set_config("sud")

corpus_file = "fr_gsd-sud-test.conllu"
gsd_test_2_14 = "https://raw.githubusercontent.com/surfacesyntacticud/SUD_French-GSD/r2.14/fr_gsd-sud-test.conllu"

if not os.path.isfile(corpus_file):
	wget.download(gsd_test_2_14)

corpus = Corpus(corpus_file)

class TestRequest(unittest.TestCase):

	def test_tuto_1(self):
		self.assertEqual(corpus.count(Request('pattern { X[] }')),10431)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB ] }')),821)
		self.assertEqual(corpus.count(Request('pattern { X [ lemma="," ] }')),489)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB, Mood=Ind, Person="1" ] }')),28)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB, lemma="être"|"avoir", Mood=Ind|Imp ] }')),39)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB, VerbForm <> Fin|Inf ] }')),286)
		self.assertEqual(corpus.count(Request('pattern { X [ form = re".*er" ] }')),168)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB, Tense ] }')),644)
		self.assertEqual(corpus.count(Request('pattern { X1 [ lemma="faire" ]; X2 [ lemma="faire" ] }')),6)

	def test_tuto_2(self):
		self.assertEqual(corpus.count(Request('pattern { X []; Y []; X -> Y; }')),10015)
		self.assertEqual(corpus.count(Request('pattern { X -> Y; }')),10015)
		self.assertEqual(corpus.count(Request('pattern { X -[subj]-> Y; }')),522)
		self.assertEqual(corpus.count(Request('pattern { X -[subj|comp:obj]-> Y }')),2573)
		self.assertEqual(corpus.count(Request('pattern { X -[^subj|comp:obj]-> Y }')),7442)
		self.assertEqual(corpus.count(Request('pattern { X -[re".*aux.*"]-> Y }')),206)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=NUM ]; X -[mod]-> Y; }')),24)
		self.assertEqual(corpus.count(Request('pattern { AP -[comp:aux@pass]-> V; V -[comp:obl@agent]-> P; P[lemma="par"]; P -[comp:obj]-> A }')),14)

	def test_tuto_3(self):
		self.assertEqual(corpus.count(Request('pattern { X -[subj]-> Y; X.Number = Y.Number }')),428)
		self.assertEqual(corpus.count(Request('pattern { X -[subj]-> Y; X.Number <> Y.Number }')),5)
		self.assertEqual(corpus.count(Request('pattern { X -[1=subj]-> Y; X << Y }')),34)
		self.assertEqual(corpus.count(Request('pattern { X1 [upos=VERB]; X2 [upos=NOUN]; X1 < X2 }')),23)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB, lemma="faire"]; * -[root]-> X; }')),4)
		self.assertEqual(corpus.count(Request('pattern { X [ upos=VERB, lemma="faire"]; X -[subj]-> *; }')),7)


	def test_tuto_4(self):
		self.assertEqual(corpus.count(Request(''' pattern { X [ upos=VERB] }
                                              without { X -[1=subj]-> S } ''')),493)
		self.assertEqual(corpus.count(Request(''' pattern { X1 [upos=DET]; X2 [upos=NOUN]; X1 < X2 }
                                              without { X2 -[det]-> X1} ''')),8)
		self.assertEqual(corpus.count(Request(''' pattern { X [ upos=VERB, lemma="faire"] }
																							without { V -[1=subj]-> S }
																							without { V -[comp:obj]-> O } ''')),0)
		self.assertEqual(corpus.count(Request(''' pattern { X [ upos=VERB] }
																							without { X -[1=subj]-> S }
																							without { X [VerbForm = Ger|Inf|Part] }
																							without { C -[1=conj,2=coord]-> X }
																							without { X [Mood = Imp] } ''')),3)

	def test_tuto_5(self):
		clust1 = corpus.count(Request('pattern { X [form="son"] }'),clustering_keys=['X.upos'])
		self.assertEqual(len(clust1), 2)
		self.assertEqual(clust1['DET'], 32)
		self.assertEqual(clust1['NOUN'], 1)

		clust2 = corpus.count(Request('pattern { X -> Y; Y [upos=NOUN] }'),clustering_keys=['X.upos'])
		self.assertEqual(len(clust2), 14)
		self.assertEqual(clust2['ADP'], 1004)
		self.assertEqual(clust2['VERB'], 407)

		clust3 = corpus.count(Request('pattern { X [upos=AUX] }'),clustering_keys=['X.lemma'])
		self.assertEqual(len(clust3), 3)
		self.assertEqual(clust3['faire'], 13)

		clust4 = corpus.count(Request('pattern { X[upos = VERB|AUX] }'),clustering_keys=['X.VerbForm'])
		self.assertEqual(len(clust4), 4)
		self.assertEqual(clust4['__undefined__'], 2)

		clust5 = corpus.count(Request('pattern { e: X -> Y; X[upos=VERB]; Y[upos=NOUN] }'),clustering_keys=['e.label'])
		self.assertEqual(len(clust5), 9)
		self.assertEqual(clust5['subj@pass'], 3)


	def test_misc(self):
		self.assertEqual(corpus.count(Request('pattern { } % with comment')),416)
		self.assertEqual(corpus.count(Request('')),416)
		self.assertEqual(corpus.count(Request('pattern { X[form = /le/i] }')),368)
		self.assertEqual(corpus.count(Request('pattern { X [upos=VERB, VerbForm=Part, Tense=Past]|[upos=ADJ] }')),858)
		self.assertEqual(corpus.count(Request(''' pattern { V1 [upos=VERB]; V1 ->> P; P[upos=PRON, PronType=Rel] }
                                              without { V2 [upos=VERB]; V1 ->> V2; V2 ->> P; }''')), 66)
		self.assertEqual(corpus.count(Request('pattern { e1: N1 -> M1; e2: N2 -> M2; e1 >< e2 }')),306)
		self.assertEqual(corpus.count(Request('pattern { e1: N1 -> M1; e2: N2 -[det]-> M2; e1 << e2 }')),13)
		self.assertEqual(corpus.count(Request('pattern { e1: N1 -> M1; e2: N2 -> M2; M1 < M2; e1 <> e2 }')),2222)
		self.assertEqual(corpus.count(Request('pattern { e: N -[det]-> M; X << e; X[upos = CCONJ] }')),5)
		self.assertEqual(corpus.count(Request('pattern { X1 [upos=SCONJ]; X2 [upos=SCONJ]; X1 << X2; length(X1,X2) = 4 }')),1)
		self.assertEqual(corpus.count(Request('pattern { X1 [upos=SCONJ]; X2 [upos=SCONJ]; X1 << X2; delta(X1,X2) < -1 }')),0)
		self.assertEqual(corpus.count(Request('pattern { X1 [ lemma="faire" ]; X2$ [ lemma="faire" ] }')),41)
		self.assertEqual(corpus.count(Request('pattern { X -[1=comp, 2=obl|aux]-> Y }')),503)
		self.assertEqual(corpus.count(Request('pattern { X -[1=comp, 2=*]-> Y }')),2778)
		self.assertEqual(corpus.count(Request('global { is_not_projective }')),46)
		self.assertEqual(corpus.count(Request('global { is_projective }')),370)
		self.assertEqual(corpus.count(Request('global { is_cyclic }')),0)
		self.assertEqual(corpus.count(Request('global { text = re".*\\baux\\b.*" }')),20)
		self.assertEqual(corpus.count(Request('pattern { X1 [ lemma="faire" ]; X2 [ lemma="faire" ]; X1.__id__ < X2.__id__ }')),3)

if __name__ == '__main__':
    unittest.main()
