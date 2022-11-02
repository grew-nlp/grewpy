
import unittest
import sys, os
import json, re

sys.path.insert(0, os.path.abspath("./grewpy"))  # Use local grew lib
import grew

graph = grew.Corpus("examples/resources/test1.conllu")[0]

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
        grs = grew.GRS("examples/resources/test1.grs")
        self.assertGreaterEqual(grs.index, 1, "could not load the file test1.conllu")
        self.assertEqual(len(grs.run(graph, strat="s1")),2, "grs fail on strategy 's1'" )
        self.assertEqual(len(grs.run(graph, strat="s2")),1)
        self.assertEqual(len(grs.run(graph, strat="s3")), 1)

    def test_read_string(self): 
        grs = grew.GRS(TestGRS.string_grs)
        self.assertGreaterEqual(grs.index, 1, "could not load the string")
        self.assertEqual(len(grs.run(graph, strat="s1")),
                         2, "grs fail on strategy 's1'")
        self.assertEqual(len(grs.run(graph, strat="s2")), 1)
        self.assertEqual(len(grs.run(graph, strat="s3")), 1)

    def test_explicit_building(self):
        req_det_n = grew.Request("N1[upos=DET]", "N2[upos=NOUN]; N1 < N2").without("N2 -> N1")
        add_det_cde = grew.Command("add_edge N2 -[det]-> N1")
        R = grew.Rule(req_det_n, add_det_cde)
        grs = grew.GRS({"det": R, "s1": "det", "s2": "Onf (det)", "s3": "Iter(det)"})
        self.assertEqual(len(grs.run(graph, strat="s1")),
                         2, "grs fail on strategy 's1'")
        self.assertEqual(len(grs.run(graph, strat="s2")), 1)
        self.assertEqual(len(grs.run(graph, strat="s3")), 1)
        
    def test_equivalence(self):
        grs1 = grs = grew.GRS("examples/resources/test1.grs")
        grs2 = grew.GRS(TestGRS.string_grs)
        req_det_n = grew.Request(
            "N1[upos=DET]; N2[upos=NOUN]; N1 < N2").without("N2 -> N1")
        add_det_cde = grew.Command("add_edge N2 -[det]-> N1")
        R = grew.Rule(req_det_n, add_det_cde)
        grs3 = grew.GRS(
            {"det": R, "s1": "det", "s2": "Onf (det)", "s3": "Iter (det)"})
        j1 = re.sub(r'[ "]', '',json.dumps(grs1.json_data()))
        j2 = re.sub(r'[ "]', '',json.dumps(grs2.json_data()))
        j3 = re.sub(r'[ "]', '',json.dumps(grs3.json_data()))

        #self.assertEqual(j1,j2)
        #self.assertEqual(j1,j3)


if __name__ == '__main__':
    unittest.main()
