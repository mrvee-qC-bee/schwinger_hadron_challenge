"""(b) alternative VALID solutions that should pass."""
from review_grader_common import setup, run, ROOT
import os, sys, numpy as np
ff = setup("valid")
import schwinger_reference as R
import challenge_utils as cu
sys.path.insert(0, os.path.join(ROOT, "tools"))
import test_grader as T
ff.SUBMISSION_DIR = os.path.join(ROOT, "scratch", "review_sub_valid")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_ibm_runtime.fake_provider import FakeKingston
r = ff._refs(); kb = FakeKingston()
M, G = 0.5, 0.3

# ---- ex0.5 variants of trotter_step that are unitarily identical to the pinned ordering
def rz_layer(qc, L, dt, g):
    for k in range(L // 2 - 1):
        qc.rz(g**2 * dt, 2 * k); qc.rz(0.5 * g**2 * dt, 2 * k + 1)
    qc.rz(0.5 * g**2 * dt, L - 2); qc.rz(-0.5 * g**2 * dt, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * dt, L + 2 * k); qc.rz(-g**2 * dt, L + 2 * k + 1)
def step_reversed_sublayers(qc, L, dt, m, g):
    n = 2 * L
    for j in reversed(range(1, n - 1, 2)): qc.append(R.RXXplus(dt / 4), [j, j + 1])
    for j in reversed(range(0, n - 1, 2)): qc.append(R.RXXplus(dt / 4), [j, j + 1])
    qc = cu.trotter_step_electric_2q(qc, L, dt, g)          # barbells BEFORE the Rz layer
    rz_layer(qc, L, dt, g)
    for j in range(n): qc.rz((-1) ** j * m * dt, j)
    for j in reversed(range(0, n - 1, 2)): qc.append(R.RXXplus(dt / 4), [j, j + 1])
    for j in reversed(range(1, n - 1, 2)): qc.append(R.RXXplus(dt / 4), [j, j + 1])
    return qc
def step_native_rxx_ryy(qc, L, dt, m, g):
    n = 2 * L
    def kin(js):
        for j in js: qc.rxx(dt / 2, j, j + 1); qc.ryy(dt / 2, j, j + 1)
    kin(range(1, n - 1, 2)); kin(range(0, n - 1, 2))
    for j in range(n): qc.rz((-1) ** j * m * dt, j)        # mass layer first (diagonal, commutes)
    rz_layer(qc, L, dt, g); qc = cu.trotter_step_electric_2q(qc, L, dt, g)
    kin(range(0, n - 1, 2)); kin(range(1, n - 1, 2))
    return qc
def step_mixed_sublayer_interleaved(qc, L, dt, m, g):
    """odd and even bonds interleaved in a gate order that still equals odd-layer-then-even-layer? NO -- only when the
    two layers commute per bond. Here: apply bond (1,2), then (0,1)... this is NOT equivalent; used as a control."""
    n = 2 * L
    for j in range(0, n - 1): qc.append(R.RXXplus(dt / 4), [j, j + 1])
    rz_layer(qc, L, dt, g); qc = cu.trotter_step_electric_2q(qc, L, dt, g)
    for j in range(n): qc.rz((-1) ** j * m * dt, j)
    for j in reversed(range(0, n - 1)): qc.append(R.RXXplus(dt / 4), [j, j + 1])
    return qc
run("ex0.5 reversed gate order inside each sublayer + barbells before Rz", ff.grade_ex0_5, step_reversed_sublayers, T.make_evolve(step_reversed_sublayers), R.prep_wave)
run("ex0.5 native rxx/ryy kinetic gates, mass layer first", ff.grade_ex0_5, step_native_rxx_ryy, T.make_evolve(step_native_rxx_ryy), R.prep_wave)
run("ex0.5 control: sequential bond sweep (different 2nd-order product)", ff.grade_ex0_5, step_mixed_sublayer_interleaved, T.make_evolve(step_mixed_sublayer_interleaved), R.prep_wave)
# evolve_circuits returning a list, and with n_steps computed as int(np.ceil(t)) (same for t=2,4)
def evolve_list(qc_init, L, t, m, g):
    return list(R.evolve_circuits(qc_init, L, t, m, g))
run("ex0.5 evolve_circuits returns a list", ff.grade_ex0_5, R.trotter_step, evolve_list, R.prep_wave)

# ---- ex1.1 Hamiltonian with a different grouping (unsimplified, sparse-list, no identity)
def H_alt(L, m, g):
    n = 2 * L; terms = []
    for j in range(n):
        terms.append(("Z", [j], m / 2 * (-1) ** j))
    for j in range(n - 1):
        terms += [("XX", [j, j + 1], 0.25), ("YY", [j, j + 1], 0.25)]
    # electric: (sum_{k<=j} Q_k)^2 expanded by hand, Q_k = -(Z_k + (-1)^k)/2
    for j in range(n - 1):
        ks = list(range(j + 1))
        for a in ks:
            for b in ks:
                ca = -0.5; cb = -0.5
                if a == b:
                    terms.append(("I", [], g**2 / 2 * ca * cb))
                else:
                    terms.append(("ZZ", [a, b], g**2 / 2 * ca * cb))
                terms.append(("Z", [a], g**2 / 2 * ca * (-0.5 * (-1) ** b)))
                terms.append(("Z", [b], g**2 / 2 * cb * (-0.5 * (-1) ** a)))
                terms.append(("I", [], g**2 / 2 * 0.25 * (-1) ** (a + b)))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)      # NOT simplified, with identity
def Hel_alt(L, g):
    return R.electric_hamiltonian_truncated(L, g) + SparsePauliOp("I" * 2 * L, 3.0)   # extra identity
run("ex1.1 hand-expanded unsimplified H (+identity), Hel with extra identity", ff.grade_ex1_1, H_alt, Hel_alt, float(r["E0_L8"]))
run("ex1.1 E0 given as np.float32", ff.grade_ex1_1, R.schwinger_hamiltonian, R.electric_hamiltonian_truncated, np.float32(r["E0_L8"]))

# ---- ex1.4 alternative key formats
tab = {(t, dt): float(r[f"trotter_table_t{t}_dt{dt:g}"]) for t in (2, 4) for dt in (1.0, 0.5, 0.25)}
run("ex1.4 keys as 't=2, dt=0.5' strings", ff.grade_ex1_4, {f"t={k[0]:g}, dt={k[1]:g}": v for k, v in tab.items()}, float(r["richardson_error_t4"]), "Trotter (dt=1)")
run("ex1.4 keys as int tuples (2, 1)", ff.grade_ex1_4, {(int(k[0]), k[1]): v for k, v in tab.items()}, float(r["richardson_error_t4"]), "trotter")
run("ex1.4 nested dict {t: {dt: err}}", ff.grade_ex1_4, {2: {1.0: tab[(2, 1.0)], 0.5: tab[(2, 0.5)], 0.25: tab[(2, 0.25)]}, 4: {1.0: tab[(4, 1.0)], 0.5: tab[(4, 0.5)], 0.25: tab[(4, 0.25)]}}, float(r["richardson_error_t4"]), "trotter")

# ---- ex3.1 odr_uncertainty: same Gaussian MC, different seeds / RNG streams / delta method
def mc_unc(seed_used, rng_kind="default"):
    def odr_uncertainty(chi, chi_std, chi_cal, chi_cal_std, chi_exact, L, suppression_threshold=0.01, n_samples=2000, seed=0):
        s = seed if seed_used is None else seed_used
        rng = np.random.default_rng(s) if rng_kind == "default" else np.random.RandomState(s)
        chi = np.asarray(chi, float); chi_cal = np.asarray(chi_cal, float)
        samples = np.empty((n_samples, 2 * L))
        for i in range(n_samples):
            if rng_kind == "default":
                dc = rng.normal(size=2 * L); dcal = rng.normal(size=2 * L)
            else:
                dc = rng.randn(2 * L); dcal = rng.randn(2 * L)
            samples[i] = R.odr_mitigate(chi + np.asarray(chi_std) * dc, chi_cal + np.asarray(chi_cal_std) * dcal, chi_exact, L, suppression_threshold)
        return np.nanstd(samples, axis=0)
    return odr_uncertainty
# diagnose the synthetic set used by the grader
chi_true, chi_exact, f, chi_cal, chi_meas = ff._synthetic_odr(8, 3200)
rng = np.random.default_rng(3201); chi_std = rng.uniform(0.01, 0.05, 16); cal_std = rng.uniform(0.01, 0.05, 16)
print("ex3.1 uncertainty set: f =", np.round(f, 3)); print("  relative std of f (cal_std/(1-chi_exact)/f) =", np.round(cal_std / (1 - chi_exact) / f, 2))
fails = []
for s in range(12):
    p = run(f"ex3.1 MC uncertainty, default_rng seed {s} (2000 samples)", ff.grade_ex3_1, R.odr_mitigate, mc_unc(s), T.odr_bias)
    if p < 7: fails.append(s)
print("seeds losing points:", fails)
run("ex3.1 MC uncertainty, np.random.RandomState(0)", ff.grade_ex3_1, R.odr_mitigate, mc_unc(0, "legacy"), T.odr_bias)
run("ex3.1 MC uncertainty, honours seed kwarg, 20000 samples", ff.grade_ex3_1, R.odr_mitigate, lambda *a, **k: mc_unc(None)(*a, **dict(k, n_samples=20000)), T.odr_bias)

# ---- ex3.2 twirl variants
def twirl_sorted_cz(isa_circuit, seed):
    """valid twirl, but every cz is re-emitted with sorted qubit order (cz is symmetric)."""
    tc = T.twirl_circuit(isa_circuit, seed)
    out = tc.copy_empty_like()
    for inst in tc.data:
        if inst.operation.name == "cz":
            qs = sorted(inst.qubits, key=lambda q: tc.find_bit(q).index)
            out.cz(qs[0], qs[1])
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out
run("ex3.2 valid twirl with cz qubits in sorted order", ff.grade_ex3_2, twirl_sorted_cz, T.postselect_charge)
def twirl_sx_sx(isa_circuit, seed):
    """valid twirl expressing X as sx.sx and Y as sx.sx.rz(pi) (ISA)."""
    rng = np.random.default_rng(seed)
    qc = isa_circuit.copy_empty_like()
    def P(p, q):
        if p == "X": qc.sx(q); qc.sx(q)
        elif p == "Y": qc.rz(np.pi, q); qc.sx(q); qc.sx(q)
        elif p == "Z": qc.rz(np.pi, q)
    for inst in isa_circuit.data:
        if inst.operation.name == "cz":
            qa, qb = inst.qubits; pa, pb = rng.choice(["I", "X", "Y", "Z"], 2)
            P(pa, qa); P(pb, qb); qc.append(inst.operation, inst.qubits)
            pa2 = T._PAULI_MUL[(pa, "Z")] if pb in "XY" else pa; pb2 = T._PAULI_MUL[(pb, "Z")] if pa in "XY" else pb
            P(pa2, qa); P(pb2, qb)
        else:
            qc.append(inst.operation, inst.qubits, inst.clbits)
    return qc
run("ex3.2 valid twirl with X = sx.sx", ff.grade_ex3_2, twirl_sx_sx, T.postselect_charge)
from qiskit.result import Counts
run("ex3.2 postselect returns qiskit Counts object", ff.grade_ex3_2, T.twirl_circuit, lambda c, L: Counts({k: v for k, v in c.items() if k.count('1') == L}))

# ---- ex2.3 alternative DFS chains (other seeds / fewer restarts, as a participant search would)
for s, nr in ((1, 500), (7, 1000), (11, 4000)):
    run(f"ex2.3 randomized DFS seed {s}, {nr} restarts", ff.grade_ex2_3, lambda b, n_qubits=68, s=s, nr=nr: ff.baseline_chain(ff.live_target_summary(b), n_qubits, seed=s, n_restarts=nr, budget_s=30)[1], kb)

# ---- ex1.5 string keys
rd = os.path.join(ROOT, "reference_data")
Xqdc = np.loadtxt(f"{rd}/chi_wave_evolved_sim_L34_maxbond40.txt") - np.loadtxt(f"{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
run("ex1.5 string keys, list X_bond40", ff.grade_ex1_5, {"8": 0.0561, "20": 0.0029, "40": 0.0007}, {"8": 1.3, "20": 7.0, "40": 33.6}, list(Xqdc))
