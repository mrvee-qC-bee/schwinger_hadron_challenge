import sys, time, numpy as np
sys.path[:0]=['.','organizer','tools']
import schwinger_reference as R
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AE
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from qiskit.quantum_info import Statevector
import nbkit, nb_part_b
ns={}
src=nbkit.solution_view(nb_part_b.CELLS[9]["source"]); exec("import numpy as np\n"+src, ns)
twirl=ns["twirl_circuit"]
L4=4; m,g=0.5,0.3
qw=R.prep_wave(L4); qv=R.prep_vacuum_for_subtraction(L4)
qwp,qwm=R.evolve_circuits(qw,L4,2.0); qvp,qvm=R.evolve_circuits(qv,L4,2.0)
pm=generate_preset_pass_manager(optimization_level=1, basis_gates=["rz","sx","x","cz"], seed_transpiler=1)
circ=[pm.run(c) for c in (qwp,qwm,qvp,qvm)]
obs=R.chiral_condensate_observables(L4)
ex=lambda q: np.array([Statevector(q).expectation_value(o).real for o in obs])
Xe=ex(qwp)-ex(qvp); c0w,c0v=ex(qw),ex(qv)
print("X exact", np.round(Xe,3))
for p in [0.002, 0.005, 0.01]:
    nd=NoiseModel(); nd.add_all_qubit_quantum_error(depolarizing_error(p,2),"cz")
    na=NoiseModel(); na.add_all_qubit_quantum_error(amplitude_damping_error(p).tensor(amplitude_damping_error(p)),"cz")
    for nname,nm in [("dep",nd),("ad",na)]:
        est=AE(options={"backend_options":{"method":"density_matrix","noise_model":nm}})
        for ntw in [0,16]:
            t0=time.time()
            if ntw:
                pubs=[(twirl(c,seed=1000*i+s),obs) for i,c in enumerate(circ) for s in range(ntw)]
                evs=np.array([r.data.evs for r in est.run(pubs,precision=0).result()]).reshape(4,ntw,8).mean(1)
            else:
                evs=np.array([r.data.evs for r in est.run([(c,obs) for c in circ],precision=0).result()])
            Xr=evs[0]-evs[2]; Xm=R.odr_mitigate(evs[0],evs[1],c0w,L4)-R.odr_mitigate(evs[2],evs[3],c0v,L4)
            f=(1-evs[1])/(1-c0w)
            w=((-1.)**np.arange(8)*(evs[0]-1)).sum()
            print(f"p={p} {nname:3s} tw={ntw:2d}: raw rmse {np.sqrt(np.mean((Xr-Xe)**2)):.4f} odr rmse {np.sqrt(np.nanmean((Xm-Xe)**2)):.4f} f range {f.min():.2f}-{f.max():.2f} witness {w:+.4f} ({time.time()-t0:.1f}s)")
