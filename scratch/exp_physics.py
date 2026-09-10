import sys, time; sys.path[:0]=['.', 'organizer']
import schwinger_reference as R, numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import eigsh, expm_multiply
t0=time.time()
for L in [4,6,8]:
    H=R.schwinger_hamiltonian(L).to_matrix(sparse=True); w,v=eigsh(H,k=1,which='SA')
    sv=Statevector(R.prep_vacuum(L)).data
    print(L, "E0", w[0], "fid", abs(np.vdot(v[:,0],sv))**2, "gap", np.real(np.vdot(sv,H@sv))-w[0], time.time()-t0)
    Ht=R.schwinger_hamiltonian(L,truncated=True).to_matrix(sparse=True); wt,_=eigsh(Ht,k=1,which='SA'); print("  trunc E0", wt[0])
# charge
L=6; n=2*L
Q=sum((R._pauli(n,{k:'Z'},-0.5)+R._pauli(n,{},-0.5*(-1)**k)) for k in range(n)).simplify()
H=R.schwinger_hamiltonian(L); print("[H,Q] norm", np.linalg.norm((H@Q-Q@H).simplify().coeffs))
# truncation shift L=8 t=4
L=8; n=16; obs=R.chiral_condensate_observables(L)
psi0=Statevector(R.prep_wave(L)).data
Hf=R.schwinger_hamiltonian(L).to_matrix(sparse=True); Ht=R.schwinger_hamiltonian(L,truncated=True).to_matrix(sparse=True)
t0=time.time(); pf=expm_multiply(-1j*4.0*Hf, psi0); pt=expm_multiply(-1j*4.0*Ht, psi0); print("expm time", time.time()-t0)
def chi(psi): 
    s=Statevector(psi); return np.array([np.real(s.expectation_value(o)) for o in obs])
cf, ct = chi(pf), chi(pt)
print("truncation shift chi max", np.max(np.abs(cf-ct)), "fid", abs(np.vdot(pf,pt))**2)
# vacuum-subtracted version
pv0=Statevector(R.prep_vacuum(L)).data
vf=expm_multiply(-1j*4.0*Hf, pv0); vt=expm_multiply(-1j*4.0*Ht, pv0)
print("truncation shift X max", np.max(np.abs((cf-chi(vf))-(ct-chi(vt)))))
# trotter table
tab={}
for t in [2,4]:
    ex=chi(expm_multiply(-1j*t*Ht, psi0))
    for dt in [1,0.5,0.25]:
        ns=int(round(t/dt)); qc=R.prep_wave(L)
        for _ in range(ns): qc=R.trotter_step(qc,L,dt)
        t1=time.time(); c=chi(Statevector(qc).data); tab[(t,dt)]=np.max(np.abs(c-ex)); print(t,dt,tab[(t,dt)], f"{time.time()-t1:.1f}s")
    print(" ratios", tab[(t,1)]/tab[(t,0.5)], tab[(t,0.5)]/tab[(t,0.25)])
# richardson
t=4; ex=chi(expm_multiply(-1j*t*Ht, psi0))
def chi_dt(dt):
    qc=R.prep_wave(L)
    for _ in range(int(round(t/dt))): qc=R.trotter_step(qc,L,dt)
    return chi(Statevector(qc).data)
cR=(4*chi_dt(0.25)-chi_dt(0.5))/3; print("richardson err", np.max(np.abs(cR-ex)))
# electric layer identity check
for L in [4,6]:
    n=2*L; g=0.3; t=0.37
    qc=QuantumCircuit(n)
    for k in range(L//2-1): qc.rz(g**2*t,2*k); qc.rz(0.5*g**2*t,2*k+1)
    qc.rz(0.5*g**2*t,L-2); qc.rz(-0.5*g**2*t,L+1)
    for k in range(1,L//2): qc.rz(-0.5*g**2*t,L+2*k); qc.rz(-g**2*t,L+2*k+1)
    qc=R.cu.trotter_step_electric_2q(qc,L,t,g)
    Hel=R.electric_hamiltonian_truncated(L,g).to_matrix(sparse=True)
    rng=np.random.default_rng(1)
    for _ in range(3):
        psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n); psi/=np.linalg.norm(psi)
        a=Statevector(psi).evolve(qc).data; bb=expm_multiply(-1j*t*Hel, psi)
        print(L, "infidelity", 1-abs(np.vdot(a,bb))**2)
print("total", time.time()-t0)
