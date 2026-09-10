"""(a) Pass with WRONG solutions: everything here is built ONLY from files shipped in the participant kit
(reference_data/grader_refs.npz, the QDC text files, mps_reference_L34.npz, fallfest_grader.py itself) -- no physics."""
from review_grader_common import setup, run, ROOT
import os, numpy as np
ff = setup("leak")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm
r = ff._refs()

# --- ex0.3 / 0.4: initialize() the reference statevectors shipped in grader_refs.npz
import schwinger_reference as R
def prep_vacuum_leak(L, th1, th3):
    if L > 6: return R.prep_vacuum(L, th1, th3)          # the 16-qubit initialize() is timed in review_grader_leak_slow.py
    qc = QuantumCircuit(2 * L); qc.initialize(r[f"sv_vacuum_L{L}"]); return qc
def prep_wave_leak(L, th1, th3, th11, th22):
    if L > 6: return R.prep_wave(L, th1, th3, th11, th22)
    qc = QuantumCircuit(2 * L); qc.initialize(r[f"sv_wave_L{L}"]); return qc
run("ex0.3 prep_vacuum = initialize(refs sv_vacuum)", ff.grade_ex0_3, prep_vacuum_leak)
run("ex0.4 prep_wave = initialize(refs sv_wave)", ff.grade_ex0_4, prep_wave_leak)

# --- ex0.5: trotter_step = exact expm of the shipped H_trunc (no Trotter at all); physics circuit = initialize(sv_physics_t2_L6)
def H_from_refs(key_l, key_c):
    return SparsePauliOp([str(l) for l in r[key_l]], r[key_c])
def trotter_step_leak(qc, L, dt, m, g):
    H = H_from_refs(f"H_trunc_labels_L{L}", f"H_trunc_coeffs_L{L}").to_matrix()
    qc.append(UnitaryGate(expm(-1j * dt * H)), range(2 * L)); return qc
def evolve_circuits_leak(qc_init, L, t, m, g):
    qc = QuantumCircuit(2 * L); qc.initialize(r["sv_physics_t2_L6"]) if (L == 6 and t == 2.0) else None
    return qc, qc_init.copy()          # mitigation circuit = the initial state itself
run("ex0.5 real trotter_step + initialize(sv_physics_t2_L6) + identity mitig", ff.grade_ex0_5, R.trotter_step, evolve_circuits_leak, prep_wave_leak)

# --- ex1.1: Hamiltonians straight from the npz, E0 from the npz
def schwinger_hamiltonian_leak(L, m, g):
    return H_from_refs(f"H_full_labels_L{L}", f"H_full_coeffs_L{L}")
def electric_hamiltonian_truncated_leak(L, g):
    return H_from_refs(f"Hel_trunc_labels_L{L}", f"Hel_trunc_coeffs_L{L}")
run("ex1.1 H, H_el, E0 read from grader_refs.npz", ff.grade_ex1_1, schwinger_hamiltonian_leak, electric_hamiltonian_truncated_leak, float(r["E0_L8"]))

# --- ex1.2: electric_layer = UnitaryGate(expm(-i t Hel)) from the npz; truncation_shift from the npz
def electric_layer_leak(L, t, g):
    H = H_from_refs(f"Hel_trunc_labels_L{L}", f"Hel_trunc_coeffs_L{L}").to_matrix()
    qc = QuantumCircuit(2 * L); qc.append(UnitaryGate(expm(-1j * t * H)), range(2 * L)); return qc
import time; _t=time.time(); run("ex1.2 electric_layer = expm(refs Hel); truncation_shift from refs", ff.grade_ex1_2, electric_layer_leak, float(r["truncation_shift_L8_t4"])); print(f"   (expm-based electric_layer took {time.time()-_t:.0f} s in total)")

# --- ex1.3 / 1.4 / 1.5: numbers copied from the npz / QDC files
run("ex1.3 vqe dicts copied from refs", ff.grade_ex1_3, {L: float(r[f"vqe_fid_L{L}"]) for L in (4, 6, 8)}, {L: float(r[f"E_adapt_L{L}"] - r[f"E0_L{L}"]) for L in (4, 6, 8)})
tab = {(t, dt): float(r[f"trotter_table_t{t}_dt{dt:g}"]) for t in (2, 4) for dt in (1.0, 0.5, 0.25)}
run("ex1.4 table/richardson/dominant copied from refs", ff.grade_ex1_4, tab, float(r["richardson_error_t4"]), str(r["dominant_error"]))
rd = os.path.join(ROOT, "reference_data")
Xqdc = np.loadtxt(f"{rd}/chi_wave_evolved_sim_L34_maxbond40.txt") - np.loadtxt(f"{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
run("ex1.5 mps_err invented, mps_seconds {}, X_bond40 = QDC file", ff.grade_ex1_5, {8: 0.05, 20: 0.01, 40: 0.001}, {}, Xqdc)
run("ex1.5 same, X_bond40 = X_ref_t8_bd64 from refs", ff.grade_ex1_5, {8: 0.05, 20: 0.01, 40: 0.001}, {}, r["X_ref_t8_bd64"])

# --- ex2.3: call the grader's own public baseline
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez
def select_chain_leak(backend, n_qubits=68):
    return ff.baseline_chain(ff.target_summary(backend), n_qubits)[1]
run("ex2.3 select_chain = ff.baseline_chain(ff.target_summary(b)) (Kingston)", ff.grade_ex2_3, select_chain_leak, FakeKingston())
run("ex2.3 same on hidden FakeFez", ff.grade_ex2_3, select_chain_leak, FakeFez())

# --- ex4.1: 'canary' = the exact t=0 QDC files, invented job id
ew, evac = ff._qdc_t0_exact()
run("ex4.1 canary arrays = exact t=0 files, fake job", ff.grade_ex4_1, {"chi_wave": ew, "chi_vacuum": evac, "flagged_sites": [], "verdict": "pass"},
    {"job_id": "d00000000000000000000", "backend": "ibm_kingston", "usage_s": 4.0, "layout": list(range(68))})

# --- ex4.2: fabricated evs: chi = 1 - f (1 - chi_true) with chi_true = the shipped bond-64 MPS t=8 profiles, f = 0.5
mps = ff._mps(); f = 0.5 + 0.01 * np.sin(np.arange(68))
rng = np.random.default_rng(0)
cw_true, cv_true = mps["chi_wave_t8_bd64"], mps["chi_vacuum_t8_bd64"]
hr = {"chi_wave": 1 - f * (1 - cw_true) + 1e-4 * rng.normal(size=68), "chi_wave_mitig": 1 - f * (1 - ew),
      "chi_vacuum": 1 - f * (1 - cv_true) + 1e-4 * rng.normal(size=68), "chi_vacuum_mitig": 1 - f * (1 - evac)}
for k in list(hr): hr[k + "_std"] = np.full(68, 0.01)
import schwinger_reference as R
hr["X_mit"] = R.odr_mitigate(hr["chi_wave"], hr["chi_wave_mitig"], ew, 34) - R.odr_mitigate(hr["chi_vacuum"], hr["chi_vacuum_mitig"], evac, 34)
hr["sigma"] = np.full(68, 0.02)
job = {"job_id": "d00000000000000000001", "backend": "ibm_kingston", "usage_s": 33.0, "layout": list(range(68))}
run("ex4.2 fabricated evs from the shipped bond-64 MPS (f=0.5)", ff.grade_ex4_2, hr, job, R.odr_mitigate)
