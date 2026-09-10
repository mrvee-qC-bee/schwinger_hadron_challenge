import sys, time; sys.path[:0]=['.', 'organizer']
import schwinger_reference as R, numpy as np
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
b=FakeKingston()

def evolve_matched(qc_init, L, t, m, g, eps=1e-4):
    n_steps=int(2*np.ceil(t/2)); dt=t/n_steps; n=2*L
    qc=qc_init.copy(); qm=qc_init.copy()
    for _ in range(n_steps): qc=R.trotter_step(qc,L,dt,m,g)
    for _ in range(n_steps//2): qm=R.trotter_step(qm,L,dt,m,g)
    for j in range(0,n-1,2): qm.append(R.RXXplus(eps),[j,j+1])
    for _ in range(n_steps//2): qm=R.trotter_step(qm,L,-dt,m,g)
    return qc,qm

def cz_per_edge(isa):
    c=Counter()
    for inst in isa.data:
        if inst.operation.name=='cz':
            q=tuple(sorted(isa.find_bit(x).index for x in inst.qubits)); c[q]+=1
    return c

L=34; qw=R.prep_wave(L)
pm=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42)
qp,_=R.evolve_circuits(qw,L,8.0); isa=pm.run(qp); lay=isa.layout.initial_index_layout(filter_ancillas=True)
pm2=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=lay, layout_method='trivial')
isa_p=pm2.run(qp); cp=cz_per_edge(isa_p); print("physics cz", sum(cp.values()), isa_p.count_ops())
for eps in [1e-4, 1e-3, 1e-2]:
    _,qm=evolve_matched(qw,L,8.0,0.5,0.3,eps)
    isa_m=pm2.run(qm); cm=cz_per_edge(isa_m)
    diff={e:(cp[e],cm[e]) for e in set(cp)|set(cm) if cp[e]!=cm[e]}
    print(f"eps={eps}: mitig cz {sum(cm.values())} depth {R.two_qubit_depth(isa_m)} vs physics depth {R.two_qubit_depth(isa_p)}; edges differing: {len(diff)} {list(diff.items())[:5]}; ops {isa_m.count_ops()}")
# fidelity at L=6
L=6; qi=R.prep_wave(L)
for eps in [1e-4,1e-3]:
    _,qm=evolve_matched(qi,L,4.0,0.5,0.3,eps)
    f=abs(np.vdot(Statevector(qi).data, Statevector(qm).data))**2
    print(f"L=6 eps={eps} fidelity with init: {f:.12f}  1-f={1-f:.2e}")
# also check matched at L=6 on fake backend cz counts equal
qp6,_=R.evolve_circuits(qi,L,4.0); _,qm6=evolve_matched(qi,L,4.0,0.5,0.3,1e-4)
pm6=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42)
i6=pm6.run(qp6); lay6=i6.layout.initial_index_layout(filter_ancillas=True)
pm6b=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=lay6, layout_method='trivial')
a=pm6b.run(qp6); c=pm6b.run(qm6); print("L=6 t=4 cz physics/matched:", a.count_ops().get('cz'), c.count_ops().get('cz'), "per-edge equal:", cz_per_edge(a)==cz_per_edge(c))
_,qmn=R.evolve_circuits(qi,L,4.0); print("naive:", pm6b.run(qmn).count_ops().get('cz'))
# random-L check: L=4, t=2
for LL,tt in [(4,2.0),(8,2.0),(6,6.0)]:
    qi=R.prep_wave(LL); qp_,_=R.evolve_circuits(qi,LL,tt); _,qm_=evolve_matched(qi,LL,tt,0.5,0.3,1e-4)
    i_=pm6.run(qp_); l_=i_.layout.initial_index_layout(filter_ancillas=True)
    pmx=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=l_, layout_method='trivial')
    a=pmx.run(qp_); c=pmx.run(qm_); print(f"L={LL} t={tt} cz {a.count_ops().get('cz')} {c.count_ops().get('cz')} per-edge equal {cz_per_edge(a)==cz_per_edge(c)}; fid {abs(np.vdot(Statevector(qi).data, Statevector(qm_).data))**2:.10f}")
