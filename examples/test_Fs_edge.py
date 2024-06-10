import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join( os.path.dirname(__file__), "../"))) # Use local grew lib

from grewpy import Graph, CorpusDraft, Request, Corpus, request_counter 
from grewpy.graph import Fs_edge

corpus = Corpus("examples/resources/fr_pud-sud-test.conllu") # Load a corpus
sentence = corpus['n01001013']     # get a sentence form the corpus
fs_edge = sentence.sucs['2'][0][1] # Pick some Fs_edge in the graph

# There are several ways to print an fs_edge:
print (str(fs_edge))      # by default, print as a dict
print (repr(fs_edge))     # with repr, the contructor is added
print (fs_edge.compact()) # a specific method to get the compact representation

# (1) Fs_edge can be build from a dictionary
e = Fs_edge ({'1':'subj','deep':'expl'})
print ("dict     --> ", e)
print ("compact  --> ", e.compact ())

# (2) Fs_edge can be build from dictionary like string
e = Fs_edge ('1=comp,2=aux')
print ("dict     --> ", e)
print ("compact  --> ", e.compact ())

# (3) Fs_edge can be build from compact representation
e = Fs_edge ("comp:aux/m")
print ("dict     --> ", e)
print ("compact  --> ", e.compact ())