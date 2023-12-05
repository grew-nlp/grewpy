## antipatterns

### Usage : 

#### to learn the antipatterns

```sh
python3.11 antipatterns.py -f pud_sud-test_anti.req learn resources/fr_pud-sud-test.conllu
```

produces a set of anti-patterns stored within pud_sud-test_anti.req

### to test the antipatterns on a treebank

```sh
python3.11 antipatterns.py -f pud_sud-test_anti.req verify fr-gsd.train.conllu
```

(outputs a lot of antipatterns found in gsd)


### to get help about extra-parameters

```sh
python3.11 antipatterns.py --help
```
-------------------

## tests

Commands for running tests with the local grewpy instance:

```sh
PYTHONPATH=../grewpy python json_grs.py
PYTHONPATH=../grewpy python teste.py
PYTHONPATH=../grewpy python test2.py
PYTHONPATH=../grewpy python learner.py
```

Better solution?
 * https://fortierq.github.io/python-import/
