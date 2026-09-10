import sys, os, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
from collections import Counter
b=FakeKingston(); L=34; qw=R.prep_wave(L)
def perbond(isa):
    c=Counter()
    for inst in isa.data:
        if inst.operation.name=='cz':
            q=tuple(sorted(isa.find_bit(x).index for x in inst.qubits)); c[q]+=1
    return c
for t in [4.0,8.0]:
    qp,_=R.evolve_circuits(qw,L,t); _,qmp=R.evolve_circuits(qw,L,t,protect_midpoint=True); _,qm=R.evolve_circuits(qw,L,t)
    pm=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42)
    isa=pm.run(qp); lay=isa.layout.initial_index_layout(filter_ancillas=True)
    pm2=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=lay)
    for name,q in [('protected',qmp),('unprotected',qm)]:
        isam=pm2.run(q); cp=perbond(isa); cm=perbond(isam)
        diffs=[cm[k]-cp[k] for k in set(cp)|set(cm)]
        print(f"t={t} {name}: cz phys={sum(cp.values())} mitig={sum(cm.values())} per-bond diff min/max={min(diffs)}/{max(diffs)} bonds={len(cp)} | hist={Counter(diffs)}")
