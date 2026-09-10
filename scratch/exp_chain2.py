import sys, time; sys.path[:0]=['tools','.']
import nbkit, nb_part_a, numpy as np, math, types, builtins
# extract select_chain + helpers from the notebook source
src=[c['source'] for c in nb_part_a.CELLS if c['cell_type']=='code' and 'def select_chain' in c['source']][0]
src=nbkit.solution_view(src).split("n_cz_per_bond = n_cz_physics")[0]
ns={'np':np,'math':math,'time':time}; exec(src, ns)
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
for cls in (FakeKingston, FakeFez, FakeMarrakesh):
    b=cls(); t0=time.time(); ch=ns['select_chain'](b, time_budget=5.0); 
    edges={tuple(sorted(e)) for e in b.coupling_map.get_edges()}
    ok=all(tuple(sorted((x,y))) in edges for x,y in zip(ch[:-1],ch[1:])) and len(set(ch))==68
    print(b.name, f"{time.time()-t0:.1f}s valid={ok} cost={ns['chain_cost'](ch,b):.2f}")
b=FakeKingston(); ch=ns['select_chain'](b, time_budget=3.0, exclude=[125,126,50]); print("exclude ok:", not ({125,126,50}&set(ch)), len(ch))
ch40=ns['select_chain'](b, n_qubits=40, time_budget=3.0); print("n_qubits=40:", len(ch40), len(set(ch40)))
