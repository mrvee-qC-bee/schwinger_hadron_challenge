"""ex1.5 calibration: bond 8/20/40 (default & 1e-10 truncation threshold) vs organizer bond-64 at L=34, t=8."""
import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
L=34; obs=R.chiral_condensate_observables(L)
m=np.load(os.path.join(ROOT,'reference_data','mps_reference_L34.npz'))
X64=m['chi_wave_t8_bd64']-m['chi_vacuum_t8_bd64']
qw,_=R.evolve_circuits(R.prep_wave(L),L,8.0); qv,_=R.evolve_circuits(R.prep_vacuum_for_subtraction(L),L,8.0)
qw=qw.decompose(reps=3); qv=qv.decompose(reps=3)
for bd, thr in [(8,None),(20,None),(40,None),(40,1e-10),(32,None)]:
    bo={"method":"matrix_product_state","matrix_product_state_max_bond_dimension":bd}
    if thr: bo["matrix_product_state_truncation_threshold"]=thr
    est=AerEstimatorV2(options={"backend_options":bo,"run_options":{"seed_simulator":7}})
    t0=time.time(); r=est.run([(qw,obs),(qv,obs)]).result(); dt=time.time()-t0
    X=r[0].data.evs-r[1].data.evs
    print(f"bd={bd} thr={thr}: {dt:.1f}s max|X-X64|={np.max(np.abs(X-X64)):.4f} rmse={np.sqrt(np.mean((X-X64)**2)):.4f} window max={np.max(np.abs(X-X64)[25:43]):.4f}", flush=True)
