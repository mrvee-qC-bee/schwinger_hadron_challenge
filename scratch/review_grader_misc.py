"""Misc: transpiler layout as select_chain output; string-key crash in _norm_table isolated; grader_refs shipped?"""
from review_grader_common import setup, run, ROOT
import os, numpy as np
ff = setup("misc")
from qiskit_ibm_runtime.fake_provider import FakeKingston
r = ff._refs(); kb = FakeKingston()
layout = [int(q) for q in r["layout_t8_kingston_O3_seed42"]]
run("ex2.3 select_chain returns the O3 transpiler layout (no search)", ff.grade_ex2_3, lambda b, n_qubits=68: layout, kb)
try:
    print("_norm_table('t=2, dt=0.5'):", ff._norm_table({"t=2, dt=0.5": 0.1}))
except Exception as e:
    print("_norm_table string key RAISES:", type(e).__name__, e)
print("_norm_table('(2, 0.5)'):", ff._norm_table({"(2, 0.5)": 0.1}), " '2,0.5':", ff._norm_table({"2,0.5": 0.1}))
print("README lists grader_refs.npz in the participant kit:", "grader_refs.npz" in open(os.path.join(ROOT, "README.md")).read())
