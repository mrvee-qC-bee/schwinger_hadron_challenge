import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
b=FakeKingston(); tgt=b.target; L=6; t=4.0
def reduced_noise_model(backend, chain):
    tgt=backend.target; nm=NoiseModel(basis_gates=['cz','sx','x','rz','id'])
    for a,bq in zip(chain[:-1],chain[1:]):
        p=tgt['cz'].get((a,bq)) or tgt['cz'].get((bq,a)); e=depolarizing_error(p.error,2)
        nm.add_quantum_error(e,'cz',[a,bq]); nm.add_quantum_error(e,'cz',[bq,a])
    for q in chain:
        nm.add_quantum_error(depolarizing_error(tgt['sx'][(q,)].error,1),['sx','x'],[q])
        r=tgt['measure'][(q,)].error; nm.add_readout_error(ReadoutError([[1-r,r],[r,1-r]]),[q])
    return nm
chain=[52, 53, 54, 55, 59, 75, 74, 73, 79, 93, 94, 95]
qw=R.prep_wave(L); qpw,qmw=R.evolve_circuits(qw,L,t,protect_midpoint=True)
pm=generate_preset_pass_manager(optimization_level=3, backend=b, seed_transpiler=42, initial_layout=chain)
isas=[pm.run(c) for c in (qpw,qmw)]
obs=R.chiral_condensate_observables(L); obs_isa=[[o.apply_layout(c.layout) for o in obs] for c in isas]
nm=reduced_noise_model(b, chain)
for method, ro in [("density_matrix",{}),("statevector",{"shots":400,"seed_simulator":11}),("matrix_product_state",{"shots":400,"seed_simulator":11})]:
    est=AerEstimatorV2(options={"backend_options":{"noise_model":nm,"method":method},"run_options":ro})
    t0=time.time(); res=est.run([(c,o) for c,o in zip(isas,obs_isa)]).result(); dt=time.time()-t0
    print(f"{method} {ro}: {dt:.1f}s evs[:3]={np.round(res[0].data.evs[:3],4)} f[:3]={np.round(((1-res[1].data.evs)/(1-np.load(os.path.join(ROOT,'reference_data','grader_refs.npz'))['chi_wave_t0_L6']))[:3],3)}", flush=True)
