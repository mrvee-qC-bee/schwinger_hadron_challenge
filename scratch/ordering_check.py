"""MPS ordering check: pinned Fig. 8 ordering vs odd-first variant, L=34, t=8, bond 40, vs QDC bond-40 file."""
import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
import challenge_utils as cu
from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

def trotter_step_oddfirst(qc, L, dt, m=R.M_DEFAULT, g=R.G_DEFAULT):
    n=2*L
    for j in range(1,n-1,2): qc.append(R.RXXplus(dt/4),[j,j+1])
    for j in range(0,n-1,2): qc.append(R.RXXplus(dt/4),[j,j+1])
    for k in range(L//2-1):
        qc.rz(g**2*dt,2*k); qc.rz(0.5*g**2*dt,2*k+1)
    qc.rz(0.5*g**2*dt,L-2); qc.rz(-0.5*g**2*dt,L+1)
    for k in range(1,L//2):
        qc.rz(-0.5*g**2*dt,L+2*k); qc.rz(-g**2*dt,L+2*k+1)
    qc=cu.trotter_step_electric_2q(qc,L,dt,g)
    for j in range(n): qc.rz((-1)**j*m*dt,j)
    for j in range(0,n-1,2): qc.append(R.RXXplus(dt/4),[j,j+1])
    for j in range(1,n-1,2): qc.append(R.RXXplus(dt/4),[j,j+1])
    return qc

L=34; t=8.0; n_steps=8; dt=t/n_steps
obs=R.chiral_condensate_observables(L)
rd=os.path.join(ROOT,'reference_data')
Xq=np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt')-np.loadtxt(f'{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt')
est=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state","matrix_product_state_max_bond_dimension":40},"run_options":{"seed_simulator":7}})
out={}
for name, step in [('pinned', R.trotter_step), ('oddfirst', trotter_step_oddfirst)]:
    qw=R.prep_wave(L); qv=R.prep_vacuum_for_subtraction(L)
    for _ in range(n_steps):
        qw=step(qw,L,dt); qv=step(qv,L,dt)
    t0=time.time()
    r=est.run([(qw.decompose(reps=3),obs),(qv.decompose(reps=3),obs)]).result()
    X=r[0].data.evs-r[1].data.evs
    out[name]=X
    dev=X-Xq
    print(f"{name}: {time.time()-t0:.1f}s  max|dev|={np.max(np.abs(dev)):.4f} rmse={np.sqrt(np.mean(dev**2)):.4f}  centre={np.round(X[31:37],3)}", flush=True)
print("pinned vs oddfirst max|diff| =", np.max(np.abs(out['pinned']-out['oddfirst'])))
np.savez(os.path.join(ROOT,'scratch','ordering_check.npz'), **out, Xq=Xq)
