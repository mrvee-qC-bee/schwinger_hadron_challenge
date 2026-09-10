import sys, time; sys.path[:0]=['.', 'organizer']
import schwinger_reference as R, numpy as np
from collections import Counter
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
b=FakeKingston()
def evolve_matched(qc_init, L, t, m, g, eps=1e-4, parity=1):
    n_steps=int(2*np.ceil(t/2)); dt=t/n_steps; n=2*L
    qc=qc_init.copy(); qm=qc_init.copy()
    for _ in range(n_steps): qc=R.trotter_step(qc,L,dt,m,g)
    for _ in range(n_steps//2): qm=R.trotter_step(qm,L,dt,m,g)
    for j in range(parity,n-1,2): qm.append(R.RXXplus(eps),[j,j+1])
    for _ in range(n_steps//2): qm=R.trotter_step(qm,L,-dt,m,g)
    return qc,qm
def cz_per_edge(isa):
    c=Counter()
    for inst in isa.data:
        if inst.operation.name=='cz': c[tuple(sorted(isa.find_bit(x).index for x in inst.qubits))]+=1
    return c
def n2q(qc): return sum(1 for i in qc.decompose(reps=4).data if len(i.qubits)>1)
L=34; qw=R.prep_wave(L)
pm=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42)
qp,qmn=R.evolve_circuits(qw,L,8.0); isa=pm.run(qp); lay=isa.layout.initial_index_layout(filter_ancillas=True)
print("layout", lay); print("n2q logical", n2q(qp), "cz", isa.count_ops().get('cz'), "depth", R.two_qubit_depth(isa))
pm2=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=lay, layout_method='trivial')
isa_p=pm2.run(qp); cp=cz_per_edge(isa_p); print("physics pinned cz", sum(cp.values()))
print("naive mitig cz", pm2.run(qmn).count_ops().get('cz'))
_,qmb=R.evolve_circuits(qw,L,8.0,protect_midpoint=True); print("barrier mitig cz", pm2.run(qmb).count_ops().get('cz'))
for parity in [0,1]:
    _,qm=evolve_matched(qw,L,8.0,0.5,0.3,1e-4,parity); cm=cz_per_edge(pm2.run(qm))
    print(f"eps on parity {parity} bonds: cz {sum(cm.values())} per-edge equal {cp==cm}")
L=6; qi=R.prep_wave(L)
for parity in [0,1]:
    _,qm=evolve_matched(qi,L,4.0,0.5,0.3,1e-4,parity)
    print("L6 fid", parity, abs(np.vdot(Statevector(qi).data, Statevector(qm).data))**2)
for LL,tt in [(4,2.0),(6,4.0),(8,2.0)]:
    qi=R.prep_wave(LL); qp_,qn_=R.evolve_circuits(qi,LL,tt); _,qm_=evolve_matched(qi,LL,tt,0.5,0.3,1e-4,1)
    i_=pm.run(qp_); l_=i_.layout.initial_index_layout(filter_ancillas=True)
    pmx=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=l_, layout_method='trivial')
    a=pmx.run(qp_); c=pmx.run(qm_); print(f"L={LL} t={tt} cz phys {a.count_ops().get('cz')} matched {c.count_ops().get('cz')} naive {pmx.run(qn_).count_ops().get('cz')} per-edge equal {cz_per_edge(a)==cz_per_edge(c)}")
