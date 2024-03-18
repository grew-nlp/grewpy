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


## to get the tree, and the patterns on a treebank

````
python3 antipatterns.py -b --dependency comp:obj --export_tree test.dot -t 0.001 -f f_gsd.req learn ../fr_gsd-sud-dev.conllu
dot -Tpdf test.dot -o test.pdf
```

- where -b means the patterns are not simplified (pattern follow the decision tree)
- where --dependency will focus on some dependency
- where --export_tree will output the tree in dot,
- where -t 0.001 gives a purity threshold of nodes
- where -f outputs the request list

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
