# On going work for a new version of Python binding on Grew

## Grew code
To set up the Grew part of the code, you should run:

### Fist time
```
opam pin add libcaml-grew git+https://gitlab.inria.fr/grew/libcaml-grew#python --no-action
opam pin add grewpy git+https://gitlab.inria.fr/grew/python#python --no-action
opam install grewpy
```

### Later updates
```
opam update
opam upgrade
```

