"""Organizer tool: generate MPS reference profiles for L=34 at several times / bond dims."""
import sys, os, time, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
L=34; obs=R.chiral_condensate_observables(L)
qw=R.prep_wave(L); qv=R.prep_vacuum_for_subtraction(L)
out={}
meta={}
for bd in [64, 128]:
    est=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state","matrix_product_state_max_bond_dimension":bd,"matrix_product_state_truncation_threshold":1e-10},"run_options":{"seed_simulator":7}})
    for t in [2,4,6,8]:
        qpw,_=R.evolve_circuits(qw,L,float(t)); qpv,_=R.evolve_circuits(qv,L,float(t))
        t0=time.time()
        r=est.run([(qpw.decompose(reps=3),obs),(qpv.decompose(reps=3),obs)]).result()
        dt=time.time()-t0
        out[f"chi_wave_t{t}_bd{bd}"]=r[0].data.evs; out[f"chi_vacuum_t{t}_bd{bd}"]=r[1].data.evs
        meta[f"t{t}_bd{bd}"]={"seconds":round(dt,1)}
        print(f"t={t} bd={bd}: {dt:.1f}s  center X = {np.round((r[0].data.evs-r[1].data.evs)[32:36],3)}", flush=True)
np.savez(os.path.join(ROOT,'reference_data','mps_reference_L34.npz'), **out)
json.dump(meta, open(os.path.join(ROOT,'reference_data','mps_reference_L34_meta.json'),'w'), indent=1)
# compare bd64 vs bd128 and vs QDC bd40
rd=os.path.join(ROOT,'reference_data')
Xq=np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt')-np.loadtxt(f'{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt')
X64=out['chi_wave_t8_bd64']-out['chi_vacuum_t8_bd64']; X128=out['chi_wave_t8_bd128']-out['chi_vacuum_t8_bd128']
print("t=8: max|X64-X128| =", np.max(np.abs(X64-X128)), " max|Xqdc40-X128| =", np.max(np.abs(Xq-X128)))
print("DONE")
