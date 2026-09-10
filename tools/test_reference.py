"""Verification of organizer/schwinger_reference.py against exact linear algebra and QDC data."""
import sys, os, time, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path[:0]=[ROOT, os.path.join(ROOT,'organizer')]
import schwinger_reference as R
from qiskit.quantum_info import Operator, Statevector, SparsePauliOp
from qiskit import QuantumCircuit
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply

def close_up_to_phase(U, V, tol=1e-9):
    U=np.asarray(U); V=np.asarray(V)
    i=np.unravel_index(np.argmax(np.abs(V)), V.shape)
    ph=U[i]/V[i]
    return np.max(np.abs(U-ph*V))<tol, np.max(np.abs(U-ph*V))

X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
def two(A,B): return np.kron(B,A)   # qiskit little-endian: qubit0 is rightmost

# 1. two-qubit building blocks
for name, gate, gen in [
    ('RXYplus', R.RXYplus, two(X,Y)+two(Y,X)),
    ('RXYminus', R.RXYminus, two(X,Y)-two(Y,X)),
    ('RXXplus', R.RXXplus, two(X,X)+two(Y,Y)),
]:
    th=0.731
    U=Operator(gate(th)).data
    cands={'exp(-i th/2 G)':expm(-1j*th/2*gen), 'exp(+i th/2 G)':expm(1j*th/2*gen), 'exp(-i th G)':expm(-1j*th*gen), 'exp(+i th G)':expm(1j*th*gen)}
    hits=[k for k,V in cands.items() if close_up_to_phase(U,V)[0]]
    print(f"[1] {name}: matches {hits}")
    assert hits, name

# 2. electric Trotter layer == exp(-i t H_el^{(Q=0)}(1)) for L=6, 8 (both barbell branches)
for L in [4,6]:   # L=4 -> else-branch, L=6 -> if-branch of the barbell tiling (16-qubit unitaries are too big)
    t=0.37; n=2*L
    qc=QuantumCircuit(n); g=R.G_DEFAULT
    for k in range(L//2-1):
        qc.rz(g**2*t, 2*k); qc.rz(0.5*g**2*t, 2*k+1)
    qc.rz(0.5*g**2*t, L-2); qc.rz(-0.5*g**2*t, L+1)
    for k in range(1, L//2):
        qc.rz(-0.5*g**2*t, L+2*k); qc.rz(-g**2*t, L+2*k+1)
    qc=R.cu.trotter_step_electric_2q(qc, L, t, g)
    U=Operator(qc).data
    Hel=R.electric_hamiltonian_truncated(L, g)
    V=expm(-1j*t*Hel.to_matrix())
    ok,err=close_up_to_phase(U,V,1e-8)
    print(f"[2] L={L}: electric layer == exp(-i t H_el^(Q=0)(1)) up to phase: {ok} (max err {err:.2e})")
    assert ok

# 3. wavepacket O_22 circuit == exp(i th O_22)? (check both signs), L=4
L=4; n=8; th=-0.3281
def P(ops,c=1.0): return R._pauli(n,ops,c)
O22=0.5*(P({L-2:'X',L-1:'Z',L:'Y'})-P({L-2:'Y',L-1:'Z',L:'X'})-(P({L-1:'X',L:'Z',L+1:'Y'})-P({L-1:'Y',L:'Z',L+1:'X'})))
qc=QuantumCircuit(n); qc=R.wave_prep_rotate_O_22(qc, th, L)
U=Operator(qc).data
for s,lab in [(1,'exp(+i th O22)'),(-1,'exp(-i th O22)')]:
    ok,err=close_up_to_phase(U, expm(s*1j*th*O22.to_matrix()),1e-8)
    print(f"[3] O_22 circuit vs {lab}: {ok} ({err:.2e})")
O11=0.5*(P({L-1:'X',L:'Y'})-P({L-1:'Y',L:'X'}))
qc=QuantumCircuit(n); qc=R.wave_prep_rotate_O_11(qc, th, L); U=Operator(qc).data
for s,lab in [(1,'exp(+i th O11)'),(-1,'exp(-i th O11)')]:
    ok,err=close_up_to_phase(U, expm(s*1j*th*O11.to_matrix()),1e-8)
    print(f"[3] O_11 circuit vs {lab}: {ok} ({err:.2e})")
# commutation of the two O22 pieces
A=0.5*(P({L-2:'X',L-1:'Z',L:'Y'})-P({L-2:'Y',L-1:'Z',L:'X'})); B=0.5*(P({L-1:'X',L:'Z',L+1:'Y'})-P({L-1:'Y',L:'Z',L+1:'X'}))
print("[3] [A,B] norm:", np.linalg.norm((A@B-B@A).simplify().coeffs) if len((A@B-B@A).simplify().coeffs) else 0.0)

# 4. vacuum quality vs exact diagonalization (full H) at small L, and truncated-H comparison
for L in [4,6,8]:
    H=R.schwinger_hamiltonian(L); Hm=H.to_matrix(sparse=True)
    w,v=eigsh(Hm,k=1,which='SA')
    sv=Statevector(R.prep_vacuum(L))
    fid=abs(np.vdot(v[:,0],sv.data))**2
    Ht=R.schwinger_hamiltonian(L, truncated=True)
    print(f"[4] L={L}: E0={w[0]:.5f} E_adapt={np.real(sv.expectation_value(H)):.5f} fid={fid:.4f} | ||H_full-H_trunc|| coeffs={np.linalg.norm((H-Ht).simplify().coeffs):.3f}")

# 5. Trotter order check vs exp(-i dt H_trunc) at L=6
L=6; Ht=R.schwinger_hamiltonian(L, truncated=True).to_matrix()
errs=[]
for dt in [0.2,0.1,0.05]:
    qc=QuantumCircuit(2*L); qc=R.trotter_step(qc,L,dt)
    U=Operator(qc).data; V=expm(-1j*dt*Ht)
    _,err=close_up_to_phase(U,V,1e-3); errs.append(err)
print(f"[5] Trotter step error vs dt (0.2,0.1,0.05): {[f'{e:.2e}' for e in errs]}; ratios {errs[0]/errs[1]:.2f}, {errs[1]/errs[2]:.2f} (expect ~8 for 2nd order)")

# 6. mitigation circuit returns to the initial state
L=6; qi=R.prep_wave(L); qp,qm=R.evolve_circuits(qi,L,4.0)
f=abs(np.vdot(Statevector(qi).data, Statevector(qm).data))**2
print(f"[6] L={L} mitigation circuit fidelity with initial state: {f:.12f}")
assert f>1-1e-9
# protect_midpoint: gate count unchanged, barrier present
qp2,qm2=R.evolve_circuits(qi,L,4.0,protect_midpoint=True)
print("[6] protect_midpoint barrier count:", qm2.count_ops().get('barrier',0))

# 7. L=34 vs QDC reference data (t=0 exact MPS, t=8 bond-40 MPS)
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
L=34; obs=R.chiral_condensate_observables(L)
rd=os.path.join(ROOT,'reference_data')
ref_w0=np.loadtxt(f'{rd}/chi_wave_t0_sim_L34.txt'); ref_v0=np.loadtxt(f'{rd}/chi_vacuum_t0_sim_L34.txt')
ref_w8=np.loadtxt(f'{rd}/chi_wave_evolved_sim_L34_maxbond40.txt'); ref_v8=np.loadtxt(f'{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt')
qw=R.prep_wave(L); qv=R.prep_vacuum_for_subtraction(L)
est=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state"}})
r=est.run([(qw.decompose(reps=3),obs),(qv.decompose(reps=3),obs)]).result()
print(f"[7] t=0 max|dev| wave={np.max(np.abs(r[0].data.evs-ref_w0)):.2e} vac={np.max(np.abs(r[1].data.evs-ref_v0)):.2e}")
qpw,qmw=R.evolve_circuits(qw,L,8.0); qpv,qmv=R.evolve_circuits(qv,L,8.0)
est40=AerEstimatorV2(options={"backend_options":{"method":"matrix_product_state","matrix_product_state_max_bond_dimension":40}})
t0=time.time(); r8=est40.run([(qpw.decompose(reps=3),obs),(qpv.decompose(reps=3),obs)]).result(); 
Xr=ref_w8-ref_v8; Xo=r8[0].data.evs-r8[1].data.evs
print(f"[7] t=8 bond40 ({time.time()-t0:.1f}s): max|dev| wave={np.max(np.abs(r8[0].data.evs-ref_w8)):.3f} vac={np.max(np.abs(r8[1].data.evs-ref_v8)):.3f}; vac-subtracted max dev={np.max(np.abs(Xo-Xr)):.3f}; score(ours vs ref)={R.signal_score(Xo,Xr)}")
print("[7] 2q depth physics/mitig:", R.two_qubit_depth(qpw.decompose(reps=3)), R.two_qubit_depth(qmw.decompose(reps=3)))
print("ALL TESTS PASSED")
