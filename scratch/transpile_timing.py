import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
b=FakeKingston()
L=34
qw=R.prep_wave(L)
for t in [4.0, 8.0]:
    qp,qm=R.evolve_circuits(qw,L,t)
    qpd=qp.decompose(reps=3); qmd=qm.decompose(reps=3)
    n2=sum(1 for i in qpd.data if len(i.qubits)>1)
    print(f"t={t} logical 2q physics={n2} mitig={sum(1 for i in qmd.data if len(i.qubits)>1)}", flush=True)
    for ol in [1,3]:
        t0=time.time(); pm=generate_preset_pass_manager(optimization_level=ol, backend=b, seed_transpiler=42); isa=pm.run(qp); dt=time.time()-t0
        lay=isa.layout.initial_index_layout(filter_ancillas=True)
        print(f"  O{ol}: {dt:.1f}s cz={isa.count_ops().get('cz')} depth2q={R.two_qubit_depth(isa)} layout[:6]={lay[:6]} final==init {isa.layout.final_index_layout()==lay}", flush=True)
        t0=time.time(); pm2=generate_preset_pass_manager(optimization_level=ol, backend=b, seed_transpiler=42, initial_layout=lay); isam=pm2.run(qm); dt=time.time()-t0
        print(f"  O{ol} mitig w/ layout: {dt:.1f}s cz={isam.count_ops().get('cz')}", flush=True)
        qp2,qm2=R.evolve_circuits(qw,L,t,protect_midpoint=True)
        t0=time.time(); isam2=pm2.run(qm2); dt=time.time()-t0
        print(f"  O{ol} mitig protected: {dt:.1f}s cz={isam2.count_ops().get('cz')}", flush=True)
# bonus B1 counts
for ns in [16,32]:
    qp,_=R.evolve_circuits(qw,L,8.0,n_steps=ns)
    qpd=qp.decompose(reps=3)
    print(f"n_steps={ns}: 2q={sum(1 for i in qpd.data if len(i.qubits)>1)}")
