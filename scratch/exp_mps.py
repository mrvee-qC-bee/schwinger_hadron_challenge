import sys, time, os; sys.path[:0]=['.', 'organizer']
import schwinger_reference as R, numpy as np
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
L=34; obs=R.chiral_condensate_observables(L)
qw=R.prep_wave(L); qv=R.prep_vacuum_for_subtraction(L)
qpw,_=R.evolve_circuits(qw,L,8.0); qpv,_=R.evolve_circuits(qv,L,8.0)
rd='reference_data'
Xq=np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt')-np.loadtxt(f'{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt')
res={}
for bd in [8,12,20,40]:
    est=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state","matrix_product_state_max_bond_dimension":bd,"matrix_product_state_truncation_threshold":1e-10},"run_options":{"seed_simulator":7}})
    t0=time.time(); r=est.run([(qpw.decompose(reps=3),obs),(qpv.decompose(reps=3),obs)]).result(); dt=time.time()-t0
    X=r[0].data.evs-r[1].data.evs; res[bd]=X
    print(f"bd={bd}: {dt:.1f}s  max|X-Xqdc40|={np.max(np.abs(X-Xq)):.4f} center {np.round(X[31:37],3)}", flush=True)
np.savez('scratch/mps_scan.npz', **{f"X{bd}":v for bd,v in res.items()})
