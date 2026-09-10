"""Notebook select_chain (copied from tools/nb_part_a.py cell) vs the grader's new baseline on the three fake backends."""
import sys, os, re, time, math, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, "tools")
import nb_part_a, nbkit, fallfest_grader as ff
src = next(c["source"] for c in nb_part_a.CELLS if c["cell_type"]=="code" and "def select_chain" in c["source"])
src = nbkit.solution_view(src).split("\nn_cz_per_bond = ")[0]
ns = {"np": np, "math": math, "time": time}; exec(src, ns)
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
for B in (FakeKingston, FakeFez, FakeMarrakesh):
    b = B(); summ = ff.target_summary(b)
    t0=time.time(); cb, base = ff.baseline_chain(summ); tb=time.time()-t0
    for seed in (0, 1):
        t0=time.time(); ch = ns["select_chain"](b, 68, seed=seed); t=time.time()-t0
        c = ff.chain_cost(summ, ch)
        print(f"{b.name}: baseline {cb:.4f} ({tb:.1f}s) | notebook seed {seed}: {c:.4f} ({t:.1f}s) ratio {c/cb:.3f} | nb-cost {ns['chain_cost'](ch,b):.4f} vs nb-cost(baseline) {ns['chain_cost'](base,b):.4f}")
