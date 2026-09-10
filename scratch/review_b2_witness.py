import sys, numpy as np, time
sys.path[:0]=['.','organizer']
import schwinger_reference as R
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
_CZ = {"II":"II","IX":"ZX","IY":"ZY","IZ":"IZ","XI":"XZ","XX":"YY","XY":"YX","XZ":"XI","YI":"YZ","YX":"XY","YY":"XX","YZ":"YI","ZI":"ZI","ZX":"IX","ZY":"IY","ZZ":"ZZ"}
def _ap(qc,p,q):
    if p=="X": qc.x(q)
    elif p=="Z": qc.rz(np.pi,q)
    elif p=="Y": qc.x(q); qc.rz(np.pi,q)
def twirl(isa,seed):
    rng=np.random.default_rng(seed); out=isa.copy_empty_like()
    for inst in isa.data:
        if inst.operation.name=="cz":
            q0,q1=inst.qubits; a,b="IXYZ"[rng.integers(4)],"IXYZ"[rng.integers(4)]; c,d=_CZ[a+b]
            _ap(out,a,q0);_ap(out,b,q1);out.cz(q0,q1);_ap(out,c,q0);_ap(out,d,q1)
        else: out.append(inst.operation,inst.qubits,inst.clbits)
    return out
def exact_chi(qc,L): sv=Statevector(qc); return np.array([sv.expectation_value(o).real for o in R.chiral_condensate_observables(L)])
L4=4; m,g=0.5,0.3
qw4=R.prep_wave(L4); qv4=R.prep_vacuum_for_subtraction(L4)
qw_p,qw_m=R.evolve_circuits(qw4,L4,2.0); qv_p,qv_m=R.evolve_circuits(qv4,L4,2.0)
# ---- 3.2 witness toy (O1)
pm1=generate_preset_pass_manager(optimization_level=1, basis_gates=["rz","sx","x","cz"], seed_transpiler=1)
isa4=pm1.run(qw_p)
tw4=twirl(isa4,0); print('twirl equiv', Operator(tw4).equiv(Operator(isa4)))
nd=NoiseModel(); nd.add_all_qubit_quantum_error(depolarizing_error(0.02,2),"cz"); nd.add_all_qubit_quantum_error(depolarizing_error(0.002,1),["sx","x"])
na=NoiseModel(); na.add_all_qubit_quantum_error(amplitude_damping_error(0.02).tensor(amplitude_damping_error(0.02)),"cz")
def run_counts(c,noise,shots,seed):
    c=c.copy(); c.measure_all(); return AerSimulator(noise_model=noise,seed_simulator=seed).run(c,shots=shots).result().get_counts()
def zfc(counts):
    keys=[b.replace(" ","") for b in counts]; n=len(keys[0]); z=np.zeros(n); tot=0
    for b,c in zip(keys,counts.values()):
        tot+=c
        for j in range(n): z[j]+=c*(1-2*int(b[n-1-j]))
    return z/tot
for name,noise in (("depol",nd),("ampdamp",na)):
    counts=run_counts(isa4,noise,4000,11); kept=sum(c for b,c in counts.items() if b.count("1")==L4)/4000
    ctw={}
    for s in range(8):
        for b,c in run_counts(twirl(isa4,100+s),noise,500,20+s).items(): ctw[b]=ctw.get(b,0)+c
    ktw=sum(c for b,c in ctw.items() if b.count("1")==L4)/4000
    print(f"{name}: witness {zfc(counts).sum():+.3f} untw, {zfc(ctw).sum():+.3f} tw | kept {kept:.3f}/{ktw:.3f}")
print('ideal witness', f"{zfc(run_counts(isa4,None,4000,3)).sum():+.3f}")
# ---- B2 density-matrix toy (O0)
pm0=generate_preset_pass_manager(optimization_level=0, basis_gates=["rz","sx","x","cz"], seed_transpiler=1)
cB2=[pm0.run(c) for c in (qw_p,qw_m,qv_p,qv_m)]
obs4=R.chiral_condensate_observables(L4)
Xex=exact_chi(qw_p,L4)-exact_chi(qv_p,L4); cw0,cv0=exact_chi(qw4,L4),exact_chi(qv4,L4)
P=0.005
nd2=NoiseModel(); nd2.add_all_qubit_quantum_error(depolarizing_error(P,2),"cz")
na2=NoiseModel(); na2.add_all_qubit_quantum_error(amplitude_damping_error(P).tensor(amplitude_damping_error(P)),"cz")
NT={"depolarizing":8,"amplitude_damping":32}
t0=time.time()
for nname,nm_ in (("depolarizing",nd2),("amplitude_damping",na2)):
    est=AerEstimatorV2(options={"backend_options":{"method":"density_matrix","noise_model":nm_}})
    for tw in (False,True):
        if tw:
            n=NT[nname]; pubs=[(twirl(c,1000*i+s),obs4) for i,c in enumerate(cB2) for s in range(n)]
            E=np.array([r.data.evs for r in est.run(pubs,precision=0.0).result()]).reshape(4,n,2*L4).mean(axis=1)
        else:
            E=np.array([r.data.evs for r in est.run([(c,obs4) for c in cB2],precision=0.0).result()])
        Xraw=E[0]-E[2]
        Xmit=R.odr_mitigate(E[0],E[1],cw0,L4)-R.odr_mitigate(E[2],E[3],cv0,L4)
        f=(1-E[1])/(1-cw0)
        z=(-1.0)**np.arange(2*L4)*(E[0]-1)
        print(f"{nname:18s} tw={tw}: raw {np.max(np.abs(Xraw-Xex)):.4f} -> odr {np.nanmax(np.abs(Xmit-Xex)):.4f} | witness {z.sum():+.4f} | factors {f.min():.3f}-{f.max():.3f}")
print('B2 time', round(time.time()-t0))
