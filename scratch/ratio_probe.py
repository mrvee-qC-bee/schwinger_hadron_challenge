import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
L=6; n=2*L
Ht=R.schwinger_hamiltonian(L,truncated=True).to_matrix(sparse=True)
rng=np.random.default_rng(1234)
psis=[]
for _ in range(3):
    v=rng.normal(size=2**n)+1j*rng.normal(size=2**n); psis.append(v/np.linalg.norm(v))
for dts in [(0.4,0.2,0.1),(0.2,0.1,0.05)]:
    errs=[]
    for dt in dts:
        qc=QuantumCircuit(n); qc=R.trotter_step(qc,L,dt)
        e=0
        for p in psis:
            out=Statevector(p).evolve(qc).data
            ex=expm_multiply(-1j*dt*Ht, p)
            e=max(e,np.max(np.abs(out-ex)))
        errs.append(e)
    print(dts, [f"{e:.3e}" for e in errs], "ratios", errs[0]/errs[1], errs[1]/errs[2])
# odd-first variant
import challenge_utils as cu
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
errs=[]
for dt in (0.4,0.2,0.1):
    qc=QuantumCircuit(n); qc=trotter_step_oddfirst(qc,L,dt)
    e=0
    for p in psis:
        out=Statevector(p).evolve(qc).data; ex=expm_multiply(-1j*dt*Ht, p); e=max(e,np.max(np.abs(out-ex)))
    errs.append(e)
print("oddfirst", [f"{e:.3e}" for e in errs], errs[0]/errs[1], errs[1]/errs[2])
# fidelity pinned vs oddfirst t=2 physics circuit at L=6
qi=R.prep_wave(L); qp,_=R.evolve_circuits(qi,L,2.0)
qo=qi.copy()
for _ in range(2): qo=trotter_step_oddfirst(qo,L,1.0)
f=abs(np.vdot(Statevector(qp).data, Statevector(qo).data))**2
print("fidelity pinned vs oddfirst t=2 L=6:", f)
# first-order variant (kin full then el then m)
def first_order(qc,L,dt,m=R.M_DEFAULT,g=R.G_DEFAULT):
    n=2*L
    for j in range(0,n-1,2): qc.append(R.RXXplus(dt/2),[j,j+1])
    for j in range(1,n-1,2): qc.append(R.RXXplus(dt/2),[j,j+1])
    for k in range(L//2-1):
        qc.rz(g**2*dt,2*k); qc.rz(0.5*g**2*dt,2*k+1)
    qc.rz(0.5*g**2*dt,L-2); qc.rz(-0.5*g**2*dt,L+1)
    for k in range(1,L//2):
        qc.rz(-0.5*g**2*dt,L+2*k); qc.rz(-g**2*dt,L+2*k+1)
    qc=cu.trotter_step_electric_2q(qc,L,dt,g)
    for j in range(n): qc.rz((-1)**j*m*dt,j)
    return qc
errs=[]
for dt in (0.4,0.2,0.1):
    qc=QuantumCircuit(n); qc=first_order(qc,L,dt)
    e=0
    for p in psis:
        out=Statevector(p).evolve(qc).data; ex=expm_multiply(-1j*dt*Ht, p); e=max(e,np.max(np.abs(out-ex)))
    errs.append(e)
print("firstorder", [f"{e:.3e}" for e in errs], errs[0]/errs[1], errs[1]/errs[2])
