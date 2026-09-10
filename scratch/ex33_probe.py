import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
b=FakeKingston(); tgt=b.target
L=6; t=4.0
refs=np.load(os.path.join(ROOT,'reference_data','grader_refs.npz'))
# chain: 12 physical qubits, a good path
def reduced_noise_model(backend, chain):
    tgt=backend.target; nm=NoiseModel(basis_gates=['cz','sx','x','rz','id'])
    for a,bq in zip(chain[:-1],chain[1:]):
        p=tgt['cz'].get((a,bq)) or tgt['cz'].get((bq,a))
        e=depolarizing_error(p.error,2)
        nm.add_quantum_error(e,'cz',[a,bq]); nm.add_quantum_error(e,'cz',[bq,a])
    for q in chain:
        nm.add_quantum_error(depolarizing_error(tgt['sx'][(q,)].error,1),['sx','x'],[q])
        r=tgt['measure'][(q,)].error
        nm.add_readout_error(ReadoutError([[1-r,r],[r,1-r]]),[q])
    return nm
# pick chain by transpiling
qw=R.prep_wave(L); qv=R.prep_vacuum_for_subtraction(L)
qpw,qmw=R.evolve_circuits(qw,L,t,protect_midpoint=True); qpv,qmv=R.evolve_circuits(qv,L,t,protect_midpoint=True)
pm=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42)
isa0=pm.run(qpw); chain=isa0.layout.initial_index_layout(filter_ancillas=True); print("chain", chain)
pm=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=chain)
isas=[pm.run(c) for c in (qpw,qmw,qpv,qmv)]
print("cz counts", [c.count_ops().get('cz') for c in isas])
obs=R.chiral_condensate_observables(L); obs_isa=[[o.apply_layout(c.layout) for o in obs] for c in isas]
nm=reduced_noise_model(b, chain)
d=nm.to_dict(); print("noise dict keys", d.keys(), len(d['errors'])); print(d['errors'][0]); print([e for e in d['errors'] if e['type']=='roerror'][0]); print([e for e in d['errors'] if e['type']=='qerror' and len(e['gate_qubits'][0])==1][0])
for shots in [4000]:
    est=AerEstimatorV2(options={"backend_options":{"noise_model":nm,"method":"statevector"},"run_options":{"shots":shots,"seed_simulator":11}})
    t0=time.time(); res=est.run([(c,o) for c,o in zip(isas,obs_isa)]).result(); dt=time.time()-t0
    cw,cwm,cv,cvm=[r.data.evs for r in res]
    cw_ex=refs['chi_wave_t0_L6']; cv_ex=np.array([np.real(Statevector(qv).expectation_value(o)) for o in obs])
    Xraw=cw-cv; Xmit=R.odr_mitigate(cw,cwm,cw_ex,L)-R.odr_mitigate(cv,cvm,cv_ex,L)
    Xex=refs['X_trotter_L6_t4']
    f=(1-cwm)/(1-cw_ex)
    print(f"shots={shots} {dt:.1f}s RMSE raw={np.sqrt(np.mean((Xraw-Xex)**2)):.4f} mit={np.sqrt(np.mean((Xmit-Xex)**2)):.4f} factors={np.round(f,3)} stds={np.round(res[0].data.stds[:3],4)}")
    print("Xraw", np.round(Xraw,3)); print("Xmit", np.round(Xmit,3)); print("Xex ", np.round(Xex,3))
# QPY roundtrip of the L=6 t=2 ISA circuit (O1)
qp2,_=R.evolve_circuits(qw,L,2.0)
pm1=generate_preset_pass_manager(optimization_level=1, backend=b, seed_transpiler=42)
isa2=pm1.run(qp2); print("L6 t2 O1: cz", isa2.count_ops().get('cz'), 'layout', isa2.layout.initial_index_layout(filter_ancillas=True))
import io; buf=io.BytesIO(); qpy.dump(isa2, buf); arr=np.frombuffer(buf.getvalue(),dtype=np.uint8); print("qpy bytes", arr.size)
back=qpy.load(io.BytesIO(arr.tobytes()))[0]; print("roundtrip ok", back==isa2, back.layout is not None)
# O1/O2 CZ counts at L=34 t=8
qw34=R.prep_wave(34); qp8,_=R.evolve_circuits(qw34,34,8.0)
for ol in [1,2,3]:
    for seed in [42, 7]:
        pmx=generate_preset_pass_manager(optimization_level=ol, backend=b, seed_transpiler=seed); isa=pmx.run(qp8)
        print(f"O{ol} seed{seed}: cz={isa.count_ops().get('cz')} depth2q={R.two_qubit_depth(isa)}")
