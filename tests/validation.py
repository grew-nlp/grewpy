
import unittest
import sys, os
import json, re

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
from grewpy import CorpusDraft, GRS, GRSDraft, Command, Commands, Rule, Request
import grewpy

graph = CorpusDraft("examples/resources/test1.conllu")[0]

class TestGRS(unittest.TestCase):
    string_grs = """rule det {
  pattern { N1[upos=DET]; N2[upos=NOUN]; N1 < N2 }
  without { N2 -> N1 }
  commands { add_edge N2 -[det]-> N1}
}

strat s1 { det }
strat s2 { Onf (det) }
strat s3 { Iter (det) }
"""

    def test_read_file(self):
        grs = GRS("examples/resources/test1.grs")
        self.assertEqual(len(grs.run(graph, strat="s1")), 2, "grs fail on strategy 's1'" )
        self.assertEqual(len(grs.run(graph, strat="s2")), 1)
        self.assertEqual(len(grs.run(graph, strat="s3")), 1)

    def test_read_string(self): 
        grs = GRS(TestGRS.string_grs)
        self.assertEqual(len(grs.run(graph, strat="s1")), 2, "grs fail on strategy 's1'")
        self.assertEqual(len(grs.run(graph, strat="s2")), 1)
        self.assertEqual(len(grs.run(graph, strat="s3")), 1)

    def test_explicit_building(self):
        req_det_n = Request().pattern("N1[upos=DET]; N2[upos=NOUN]; N1 < N2").without("N2 -> N1")
        add_det_cde = Command("add_edge N2 -[det]-> N1")
        commands = Commands([add_det_cde])
        R = Rule(req_det_n, commands)


        grsdraft = GRSDraft({"det": R, "s1": "det", "s2": "Onf (det)", "s3": "Iter(det)"})
        grs = GRS(grsdraft)
        self.assertEqual(len(grs.run(graph, strat="s1")), 2, "grs fail on strategy 's1'")
        self.assertEqual(len(grs.run(graph, strat="s2")), 1)
        self.assertEqual(len(grs.run(graph, strat="s3")), 1)
        
    def test_equivalence(self):
        grs1 = grs = GRS("examples/resources/test1.grs")
        grs2 = GRS(TestGRS.string_grs)
        req_det_n = Request().pattern("N1[upos=DET]; N2[upos=NOUN]; N1 < N2").without("N2 -> N1")
        add_det_cde = Command("add_edge N2 -[det]-> N1")
        commands = Commands([add_det_cde])
        R = Rule(req_det_n, commands)
        grs3_draft = GRSDraft({"det": R, "s1": "det", "s2": "Onf (det)", "s3": "Iter (det)"})
        grs3 = GRS(grs3_draft)

        j1 = grs1.json()
        j2 = grs2.json()
        j3 = grs3.json()
        del j1['filename']
        del j2['filename']
        del j3['filename']

        self.assertEqual(j1,j2)
        self.assertEqual(j2,j3)

if __name__ == '__main__':
    unittest.main()
