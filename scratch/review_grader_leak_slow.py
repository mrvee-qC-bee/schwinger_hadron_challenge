"""How long do the slow oracle-leak hacks take? (initialize(sv) at 16 qubits for ex0.3)"""
from review_grader_common import setup, run
import time, numpy as np
ff = setup("leakslow"); r = ff._refs()
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
for L in (6, 8):
    t0 = time.time(); qc = QuantumCircuit(2 * L); qc.initialize(r[f"sv_vacuum_L{L}"]); sv = Statevector(qc)
    print(f"L={L}: initialize + Statevector took {time.time()-t0:.1f} s; fidelity {abs(np.vdot(sv.data, r[f'sv_vacuum_L{L}']))**2:.6f}", flush=True)
