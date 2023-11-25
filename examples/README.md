## antipatterns

### Usage : 

#### to learn the antipatterns

    python3.11 antipatterns.py -f pud_sud-test_anti.pat learn resources/fr_pud-sud-test.conllu

produces a set of anti-patterns stored within pud_sud-test_anti.pat

### to test the antipatterns on a treebank

    python3.11 antipatterns.py -f pud_sud-test_anti.pat learn fr-gsd.train.conllu

(outputs a lot of antipatterns found in gsd)


### to get help about extra-parameters

    python3.11 antipatterns.py --help

-------------------

## tests

Commands for running tests with the local grewpy instance:

```
PYTHONPATH=../grewpy python json_grs.py
PYTHONPATH=../grewpy python teste.py
PYTHONPATH=../grewpy python test2.py
PYTHONPATH=../grewpy python learner.py
```

Better solution?
 * https://fortierq.github.io/python-import/
