# Recompile the Caml code

From the top dir `python` run

```
make clean
make
```

The new compiled file is `_build/src_ocaml/grewpy.native`. Copy it with name `grewpy` somewhere in your PATH with higher precedence than the `grewpy` from opam (`.opam/…/bin/grewpy`).

The installation date can be checked with:

```
ls -l `which grewpy`
```