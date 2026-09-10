"""Recompute the t=8 bond-64 reference with the (re-pinned, odd-first) schwinger_reference.trotter_step and re-score the cached Kingston data."""
import sys, os, time, json, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
L=34; obs=R.chiral_condensate_observables(L); rd=os.path.join(ROOT,'reference_data')
qw=R.prep_wave(L); qv=R.prep_vacuum_for_subtraction(L)
qpw,_=R.evolve_circuits(qw,L,8.0); qpv,_=R.evolve_circuits(qv,L,8.0)
est=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state","matrix_product_state_max_bond_dimension":64,"matrix_product_state_truncation_threshold":1e-10},"run_options":{"seed_simulator":7}})
t0=time.time(); r=est.run([(qpw.decompose(reps=3),obs),(qpv.decompose(reps=3),obs)]).result(); print(f"bd64 t=8: {time.time()-t0:.0f}s", flush=True)
cw=r[0].data.evs; cv=r[1].data.evs; Xref=cw-cv
np.savez(os.path.join(ROOT,'scratch','xref_oddfirst_bd64_t8.npz'), chi_wave_t8_bd64=cw, chi_vacuum_t8_bd64=cv)
Xq=np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt')-np.loadtxt(f'{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt')
W=np.arange(25,43)
def rmse_w(X): return float(np.sqrt(np.nanmean((X[W]-Xref[W])**2)))
def contrast(X): return float(np.mean(X[[31,32,35,36]])-np.mean(X[[33,34]]))
def quiet(X):
    mask=np.ones(2*L,bool); mask[W]=False; return float(np.nanmedian(np.abs(X[mask])))
print('X_ref sites 31..36', np.round(Xref[31:37],4), 'RMSE_0', rmse_w(np.zeros(2*L)), 'C_ref', contrast(Xref))
print('vs QDC bd40: max|dX|', np.max(np.abs(Xref-Xq)), 'RMSE_W', rmse_w(Xq), 'chi max dev', np.max(np.abs(cw-np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt'))))
old=np.load(f'{rd}/mps_reference_L34.npz'); Xold=old['chi_wave_t8_bd64']-old['chi_vacuum_t8_bd64']
print('vs even-first bd64: max|dX|', np.max(np.abs(Xref-Xold)))
h=np.load(f'{rd}/hardware_ibm_kingston_2026-07-25.npz')
w0=np.loadtxt(f'{rd}/chi_wave_t0_sim_L34.txt'); v0=np.loadtxt(f'{rd}/chi_vacuum_t0_sim_L34.txt')
sgn=np.array([(-1)**j for j in range(2*L)]); chi=lambda z: sgn*z+1
for strat in ['odr','twirl_dd','twirl_only','dd_only','baseline','trex','zne_fold']:
    zw=h[f'{strat}__T8__z_wave']; zv=h[f'{strat}__T8__z_vacuum']; zmw=h[f'{strat}__T8__z_mitig_wave']; zmv=h[f'{strat}__T8__z_mitig_vacuum']
    Xraw=R.cp_symmetrize(chi(zw))-R.cp_symmetrize(chi(zv))
    Xmit=R.odr_mitigate(chi(zw),chi(zmw),w0,L)-R.odr_mitigate(chi(zv),chi(zmv),v0,L)
    fw=(1-chi(zmw))/(1-w0)
    pts=10*np.clip(1-rmse_w(Xmit)/0.25,0,1)+4*np.clip(1-abs(contrast(Xmit)-0.515)/0.35,0,1)+2*((contrast(Xmit)>=0.2) and (np.min(Xmit[[31,32,35,36]])-np.max(Xmit[[33,34]])>=0.25))+2*(quiet(Xmit)<=0.05)
    print(f'{strat:10s}: raw RMSE_W={rmse_w(Xraw):.3f} C={contrast(Xraw):.3f} | ODR RMSE_W={rmse_w(Xmit):.3f} C={contrast(Xmit):.3f} quiet={quiet(Xmit):.3f} pts={pts:.1f} retention W min/med/max={np.nanmin(fw[W]):.2f}/{np.nanmedian(fw[W]):.2f}/{np.nanmax(fw[W]):.2f} centre={np.round(Xmit[31:37],3)}', flush=True)
print('DONE')
