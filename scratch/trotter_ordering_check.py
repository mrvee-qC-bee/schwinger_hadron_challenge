"""Compare pinned (even-first) vs alternative (odd-first) kinetic sublattice ordering at L=34, t=8, bond 40 vs QDC bond-40 file."""
import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
import challenge_utils as cu

def trotter_step_alt(qc, L, dt, m=R.M_DEFAULT, g=R.G_DEFAULT):
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

L=34; t=8.0; obs=R.chiral_condensate_observables(L)
rd=os.path.join(ROOT,'reference_data')
Xq=np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt')-np.loadtxt(f'{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt')
est=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state","matrix_product_state_max_bond_dimension":40},"run_options":{"seed_simulator":7}})
for name, step in [('pinned even-first', R.trotter_step), ('alternative odd-first', trotter_step_alt)]:
    outs=[]
    for init in [R.prep_wave(L), R.prep_vacuum_for_subtraction(L)]:
        qc=init.copy()
        for _ in range(8): qc=step(qc,L,1.0)
        outs.append(qc.decompose(reps=3))
    t0=time.time(); r=est.run([(outs[0],obs),(outs[1],obs)]).result()
    X=r[0].data.evs-r[1].data.evs
    print(f"{name}: {time.time()-t0:.0f}s  max|X-Xqdc40|={np.max(np.abs(X-Xq)):.4f}  nrmse={np.linalg.norm(X-Xq)/np.linalg.norm(Xq):.4f}  centre={np.round(X[31:37],3)}", flush=True)
