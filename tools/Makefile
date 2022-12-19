self_doc:
	@echo "make grew_callgraph"
	@echo "make learner_callgraph"
	@echo "make clean"

grew_callgraph:
	pycg ../grew/*.py > grew_callgraph.json
	python3 cg2dot.py grew_callgraph.json grew_callgraph.dot
	dot -Tpdf -o grew_callgraph.pdf grew_callgraph.dot
	
learner_callgraph:
	pycg ../examples/learner.py > learner_callgraph.json
	python3 cg2dot.py learner_callgraph.json learner_callgraph.dot
	dot -Tpdf -o learner_callgraph.pdf learner_callgraph.dot

clean:
	rm -f learner_callgraph.* grew_callgraph.*
