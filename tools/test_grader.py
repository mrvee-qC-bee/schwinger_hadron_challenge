"""
Organizer test of fallfest_grader.py.

  (a) a "perfect participant" (organizer/schwinger_reference.py + small wrappers written here for the
      functions the reference module does not provide) must get full marks on every autograded exercise
      (ex 4.x use the cached ibm_kingston / ibm_boston data and are reported, not asserted);
  (b) wrong / hard-coded variants must lose points with helpful messages.

    python tools/test_grader.py            # ~3-4 min (the ex 3.3 noisy rehearsal is cached in scratch/)
    python tools/test_grader.py --no-cache

Scores are written to scratch/submission_test/ (FF_SUBMISSION_DIR), never to submission/.
"""
from __future__ import annotations

import io
import copy
import json
import os
import sys
import time
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(ROOT, "scratch", "submission_test"), exist_ok=True)
os.environ["FF_SUBMISSION_DIR"] = os.path.join(ROOT, "scratch", "submission_test")
sys.path[:0] = [ROOT, os.path.join(ROOT, "organizer")]

import numpy as np  # noqa: E402
import schwinger_reference as R  # noqa: E402
import challenge_utils as cu  # noqa: E402
import fallfest_grader as ff  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez  # noqa: E402
from qiskit_ibm_runtime.options import EstimatorOptions  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

NO_CACHE = "--no-cache" in sys.argv
T_START = time.time()
REFS = ff._refs()
TABLE: list[tuple[str, str, float, int, str]] = []   # (exercise, variant, points, max, detail)


def run(ex: str, variant: str, fn, *args, **kw):
    """Run a grader, capture its verdict line, record it."""
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        pts = fn(*args, **kw)
    line = buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else ""
    TABLE.append((ex, variant, float(pts), ff.MAX_POINTS[ex], f"{line[:150]} [{time.time() - t0:.1f}s]"))
    return pts


# ---------------------------------------------------------------------------
# perfect-participant wrappers (functions the reference module does not provide)
# ---------------------------------------------------------------------------
def electric_layer(L, t, g):
    qc = QuantumCircuit(2 * L)
    for k in range(L // 2 - 1):
        qc.rz(g**2 * t, 2 * k); qc.rz(0.5 * g**2 * t, 2 * k + 1)
    qc.rz(0.5 * g**2 * t, L - 2); qc.rz(-0.5 * g**2 * t, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * t, L + 2 * k); qc.rz(-g**2 * t, L + 2 * k + 1)
    return cu.trotter_step_electric_2q(qc, L, t, g)


def evolve_circuits_matched(qc_init, L, t, m, g):
    return R.evolve_circuits(qc_init, L, t, m, g, protect_midpoint=True)


def select_chain(backend, n_qubits=68):
    """Calibration-aware path search: randomized greedy DFS over -ln(1-eps) costs, 4000 restarts."""
    summ = ff.live_target_summary(backend)
    return ff.baseline_chain(summ, n_qubits, seed=1, n_restarts=4000, budget_s=30.0)[1]


def odr_uncertainty(chi, chi_std, chi_cal, chi_cal_std, chi_exact, L, suppression_threshold=0.01, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    chi = np.asarray(chi, float); chi_cal = np.asarray(chi_cal, float)
    samples = np.empty((n_samples, 2 * L))
    for s in range(n_samples):
        samples[s] = R.odr_mitigate(chi + np.asarray(chi_std) * rng.normal(size=2 * L), chi_cal + np.asarray(chi_cal_std) * rng.normal(size=2 * L),
                                    chi_exact, L, suppression_threshold)
    return np.nanstd(samples, axis=0)


def odr_bias(chi_true, f_phys, f_cal):
    chi_true = np.asarray(chi_true, float); f_phys = np.asarray(f_phys, float); f_cal = np.asarray(f_cal, float)
    return (1 - chi_true) * (1 - f_phys / f_cal)


def _set_opt(o, path, value):
    if isinstance(o, dict):
        for key in path[:-1]:
            o = o.setdefault(key, {})
        o[path[-1]] = value
    else:
        for key in path[:-1]:
            o = getattr(o, key)
        setattr(o, path[-1], value)


def mitigation_options(base, num_randomizations, shots_per_randomization, dd_sequence="XY4", readout_mitigation=False,
                       *, mutate=False, measure_twirl=True, level=0, keep_met=True):
    """Reference (and deliberately-wrong variants) of the ex 3.2 function."""
    opts = base if mutate else copy.deepcopy(base)
    fields = {("twirling", "enable_gates"): True, ("twirling", "enable_measure"): measure_twirl,
              ("twirling", "num_randomizations"): int(num_randomizations), ("twirling", "shots_per_randomization"): int(shots_per_randomization),
              ("twirling", "strategy"): "active-accum", ("dynamical_decoupling", "enable"): True,
              ("dynamical_decoupling", "sequence_type"): dd_sequence, ("resilience_level",): level,
              ("resilience", "measure_mitigation"): bool(readout_mitigation), ("resilience", "zne_mitigation"): False,
              ("resilience", "pec_mitigation"): False}
    if not keep_met:
        fields[("max_execution_time",)] = 600
    for path, value in fields.items():
        _set_opt(opts, path, value)
    return opts


def reduced_noise_model(backend, chain):
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    tgt = backend.target
    nm = NoiseModel(basis_gates=["cz", "sx", "x", "rz", "id"])
    for a, b in zip(chain[:-1], chain[1:]):
        p = tgt["cz"].get((a, b)) or tgt["cz"].get((b, a))
        e = depolarizing_error(p.error, 2)
        nm.add_quantum_error(e, "cz", [a, b]); nm.add_quantum_error(e, "cz", [b, a])
    for q in chain:
        nm.add_quantum_error(depolarizing_error(tgt["sx"][(q,)].error, 1), ["sx", "x"], [q])
        r = tgt["measure"][(q,)].error
        nm.add_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]), [q])
    return nm


def barbell_rzz(a1, a2, a3, a4, a5, a6):
    """Fractional-gate barbell: read the Z-Pauli decomposition of the diagonal CX barbell and emit rzz/rz."""
    U = Operator(cu.barbell(a1, a2, a3, a4, a5, a6)).data
    phases = np.angle(np.diag(U))
    n = 4
    # Walsh-Hadamard: phi(z) = sum_P c_P (-1)^{z.P}  ->  c_P = 1/16 sum_z phi(z) (-1)^{z.P}
    qc = QuantumCircuit(n)
    # remove global phase ambiguity by unwrapping relative to phase(0)
    d = np.exp(1j * phases); d = d / d[0]
    phi = np.angle(d)
    for mask in range(1, 16):
        c = sum(phi[z] * (-1) ** bin(z & mask).count("1") for z in range(16)) / 16
        qubits = [q for q in range(n) if mask >> q & 1]
        if abs(c) < 1e-12:
            continue
        # exp(i c Z...Z): rz(theta) = exp(-i theta/2 Z), rzz(theta) = exp(-i theta/2 ZZ)
        if len(qubits) == 1:
            qc.rz(-2 * c, qubits[0])
        elif len(qubits) == 2:
            qc.rzz(-2 * c, qubits[0], qubits[1])
        else:
            raise ValueError("barbell has 3- or 4-body Z terms?")
    return qc


# ---------------------------------------------------------------------------
# wrong variants
# ---------------------------------------------------------------------------
def trotter_step_evenfirst(qc, L, dt, m, g):
    """The other kinetic sublattice order (even bonds first): valid 2nd-order, not Fig. 8."""
    n = 2 * L
    for j in range(0, n - 1, 2):
        qc.append(R.RXXplus(dt / 4), [j, j + 1])
    for j in range(1, n - 1, 2):
        qc.append(R.RXXplus(dt / 4), [j, j + 1])
    for k in range(L // 2 - 1):
        qc.rz(g**2 * dt, 2 * k); qc.rz(0.5 * g**2 * dt, 2 * k + 1)
    qc.rz(0.5 * g**2 * dt, L - 2); qc.rz(-0.5 * g**2 * dt, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * dt, L + 2 * k); qc.rz(-g**2 * dt, L + 2 * k + 1)
    qc = cu.trotter_step_electric_2q(qc, L, dt, g)
    for j in range(n):
        qc.rz((-1) ** j * m * dt, j)
    for j in range(1, n - 1, 2):
        qc.append(R.RXXplus(dt / 4), [j, j + 1])
    for j in range(0, n - 1, 2):
        qc.append(R.RXXplus(dt / 4), [j, j + 1])
    return qc


def trotter_step_firstorder(qc, L, dt, m, g):
    n = 2 * L
    for j in range(1, n - 1, 2):
        qc.append(R.RXXplus(dt / 2), [j, j + 1])
    for j in range(0, n - 1, 2):
        qc.append(R.RXXplus(dt / 2), [j, j + 1])
    for k in range(L // 2 - 1):
        qc.rz(g**2 * dt, 2 * k); qc.rz(0.5 * g**2 * dt, 2 * k + 1)
    qc.rz(0.5 * g**2 * dt, L - 2); qc.rz(-0.5 * g**2 * dt, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * dt, L + 2 * k); qc.rz(-g**2 * dt, L + 2 * k + 1)
    qc = cu.trotter_step_electric_2q(qc, L, dt, g)
    for j in range(n):
        qc.rz((-1) ** j * m * dt, j)
    return qc


def make_evolve(step):
    def evolve(qc_init, L, t, m, g):
        n_steps = int(2 * np.ceil(t / 2)); dt = t / n_steps
        qc = qc_init.copy(); qm = qc_init.copy()
        for _ in range(n_steps):
            qc = step(qc, L, dt, m, g)
        for _ in range(n_steps // 2):
            qm = step(qm, L, dt, m, g)
        for _ in range(n_steps // 2):
            qm = step(qm, L, -dt, m, g)
        return qc, qm
    return evolve


def hamiltonian_missing_constant(L, m, g):
    """Q_k = -Z_k/2 (drops the (-1)^k I/2 part): wrong single-Z electric terms."""
    n = 2 * L
    Q = [R._pauli(n, {k: "Z"}, -0.5) for k in range(n)]
    H = R.mass_hamiltonian(L, m) + R.kinetic_hamiltonian(L)
    cum = 0 * R._pauli(n, {})
    for j in range(n - 1):
        cum = (cum + Q[j]).simplify()
        H += (g**2 / 2) * (cum @ cum)
    return H.simplify()


def prep_vacuum_wrong_interior(L, th1, th3):
    """The benchmark project's range(2, L-1) interior layer (SPEC D pitfall)."""
    qc = R.prep_strong_coupling_vacuum(L)
    qc = R.vacuum_prep_rotate_OV_1(qc, th1, L)
    for k in range(L):
        qc.append(R.RXYplus(-np.pi / 2), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(R.RXYminus(-th3), [2 * k + 1, 2 * k + 2])
    for k in range(L):
        qc.append(R.RXYplus(np.pi / 2), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(R.RXYplus(-np.pi / 2), [2 * k + 1, 2 * k + 2])
    for k in range(2, L - 1):
        qc.append(R.RXYminus(th3), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(R.RXYplus(np.pi / 2), [2 * k + 1, 2 * k + 2])
    return qc


def prep_wave_ignores_L(L, th1, th3, th11, th22):
    return R.prep_wave(8, th1, th3, th11, th22)


def odr_no_postselection(chi, chi_cal, chi_exact, L, suppression_threshold=0.01):
    f = (1 - np.asarray(chi_cal)) / (1 - np.asarray(chi_exact))
    return 1 - (1 - np.asarray(chi)) / f


def select_chain_hardcoded_kingston(backend, n_qubits=68):
    return list(REFS["layout_t8_kingston_O3_seed42"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    kb = FakeKingston()
    print("== check_env"); ff.check_env()
    for f in os.listdir(os.environ["FF_SUBMISSION_DIR"]):
        fp = os.path.join(os.environ["FF_SUBMISSION_DIR"], f)
        if os.path.isfile(fp):
            os.remove(fp)

    # ---- Part 0
    run("ex0.1", "perfect", ff.grade_ex0_1, R.chiral_condensate_observables)
    run("ex0.1", "wrong sign (-1)^(j+1)", ff.grade_ex0_1, lambda L: [SparsePauliOp.from_list([(o.paulis[0].to_label(), -o.coeffs[0]), ("I" * 2 * L, 1.0)]) for o in R.chiral_condensate_observables(L)])
    run("ex0.2", "perfect", ff.grade_ex0_2, R.RXYplus, R.RXYminus)
    run("ex0.2", "RXYminus with opposite sign", ff.grade_ex0_2, R.RXYplus, lambda th: R.RXYminus(-th))
    run("ex0.3", "perfect", ff.grade_ex0_3, R.prep_vacuum)
    run("ex0.3", "function raises", ff.grade_ex0_3, lambda L, a, b: (_ for _ in ()).throw(ValueError("boom")))
    run("ex0.3", "OV_3 interior range(2, L-1)", ff.grade_ex0_3, prep_vacuum_wrong_interior)
    run("ex0.4", "perfect", ff.grade_ex0_4, R.prep_wave)
    run("ex0.4", "prep_wave ignores L", ff.grade_ex0_4, prep_wave_ignores_L)
    run("ex0.4", "vacuum only", ff.grade_ex0_4, lambda L, a, b, c, d: R.prep_vacuum(L, a, b))
    run("ex0.5", "perfect", ff.grade_ex0_5, R.trotter_step, R.evolve_circuits, R.prep_wave)
    run("ex0.5", "even-first kinetic ordering", ff.grade_ex0_5, trotter_step_evenfirst, make_evolve(trotter_step_evenfirst), R.prep_wave)
    run("ex0.5", "first-order step", ff.grade_ex0_5, trotter_step_firstorder, make_evolve(trotter_step_firstorder), R.prep_wave)

    # ---- Part 1
    E0 = float(REFS["E0_L8"])
    run("ex1.1", "perfect", ff.grade_ex1_1, R.schwinger_hamiltonian, R.electric_hamiltonian_truncated, E0)
    run("ex1.1", "H_el without the (-1)^k I part of Q_k", ff.grade_ex1_1, hamiltonian_missing_constant, R.electric_hamiltonian_truncated, E0)
    run("ex1.1", "E0 without +m/2 I", ff.grade_ex1_1, R.schwinger_hamiltonian, R.electric_hamiltonian_truncated, E0 - 4.0)
    run("ex1.1", "truncated H passed as full", ff.grade_ex1_1, lambda L, m, g: R.schwinger_hamiltonian(L, m, g, truncated=True), R.electric_hamiltonian_truncated, E0)
    ts = float(REFS["truncation_shift_L8_t4"])
    run("ex1.2", "perfect", ff.grade_ex1_2, electric_layer, ts)
    run("ex1.2", "perfect (array form)", ff.grade_ex1_2, electric_layer, REFS["truncation_shift_array_L8_t4"])
    run("ex1.2", "Rz layer with wrong sign", ff.grade_ex1_2, lambda L, t, g: electric_layer(L, -t, g), 0.0)
    fid = {}; gap = {}
    for L in (4, 6, 8):
        H = R.schwinger_hamiltonian(L); w, v = eigsh(H.to_matrix(sparse=True), k=1, which="SA")
        sv = Statevector(R.prep_vacuum(L))
        fid[L] = float(abs(np.vdot(v[:, 0], sv.data)) ** 2); gap[L] = float(np.real(sv.expectation_value(H)) - w[0])
    run("ex1.3", "perfect (live eigsh)", ff.grade_ex1_3, fid, gap)
    run("ex1.3", "fidelity from truncated H", ff.grade_ex1_3, {k: v - 0.01 for k, v in fid.items()}, gap)
    table = {(t, dt): float(REFS[f"trotter_table_t{t}_dt{dt:g}"]) for t in (2, 4) for dt in (1.0, 0.5, 0.25)}
    run("ex1.4", "perfect", ff.grade_ex1_4, table, float(REFS["richardson_error_t4"]), "trotter")
    run("ex1.4", "first-order table (ratio 2)", ff.grade_ex1_4, {(t, dt): 0.2 * dt for t in (2, 4) for dt in (1.0, 0.5, 0.25)}, 0.02, "truncation")
    rd = os.path.join(ROOT, "reference_data")
    X40 = np.loadtxt(f"{rd}/chi_wave_evolved_sim_L34_maxbond40.txt") - np.loadtxt(f"{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
    mps_err = {8: 0.0561, 20: 0.0029, 32: 0.0013, 40: 0.0007}   # measured, see BUILD_NOTES_grader.md
    run("ex1.5", "perfect (QDC bond-40 file)", ff.grade_ex1_5, mps_err, {8: 1.3, 20: 7.0, 32: 25.3, 40: 33.6}, X40)
    Xeven = np.load(os.path.join(ROOT, "scratch", "ordering_check.npz"))["pinned"] if os.path.exists(os.path.join(ROOT, "scratch", "ordering_check.npz")) else X40 + 0.02
    run("ex1.5", "even-first bond-40 profile", ff.grade_ex1_5, mps_err, {8: 1.0, 20: 7.0, 32: 25.0, 40: 33.0}, Xeven)
    run("ex1.5", "non-monotone errors", ff.grade_ex1_5, {8: 0.05, 20: 0.03, 40: 0.04}, {8: 1, 20: 7, 40: 33}, X40)

    # ---- Part 2
    L = 34; t = 8.0
    qw = R.prep_wave(L); qv = R.prep_vacuum_for_subtraction(L)
    qpw, qmw = R.evolve_circuits(qw, L, t, protect_midpoint=True); qpv, qmv = R.evolve_circuits(qv, L, t, protect_midpoint=True)
    pm = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42)
    isa_w = pm.run(qpw); layout = isa_w.layout.initial_index_layout(filter_ancillas=True)
    pm_l = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42, initial_layout=layout)
    circuits_all_isa = [isa_w] + [pm_l.run(c) for c in (qmw, qpv, qmv)]
    obs = R.chiral_condensate_observables(L)
    observables_isa = [o.apply_layout(isa_w.layout) for o in obs]
    run("ex2.1", "perfect", ff.grade_ex2_1, circuits_all_isa, observables_isa, kb)
    pm_other = generate_preset_pass_manager(optimization_level=1, backend=kb, seed_transpiler=7)
    run("ex2.1", "4th circuit on another layout", ff.grade_ex2_1, circuits_all_isa[:3] + [pm_other.run(qmv)], observables_isa, kb)
    run("ex2.1", "logical observables (no apply_layout)", ff.grade_ex2_1, circuits_all_isa, obs, kb)
    n2q = sum(1 for i in qpw.decompose(reps=3).data if len(i.qubits) > 1)
    ncz = isa_w.count_ops().get("cz", 0)
    run("ex2.2", "perfect", ff.grade_ex2_2, n2q, ncz, evolve_circuits_matched, R.prep_wave, kb, layout)
    run("ex2.2", "cancellable junction (no barrier)", ff.grade_ex2_2, n2q, ncz, lambda qi, L, t, m, g: R.evolve_circuits(qi, L, t, m, g), R.prep_wave, kb, layout)
    run("ex2.2", "O1 CZ count", ff.grade_ex2_2, n2q, int(REFS["n_cz_physics_t8_kingston_O1_seed42"]), evolve_circuits_matched, R.prep_wave, kb, layout)
    run("ex2.3", "perfect (randomized DFS, 4000 restarts)", ff.grade_ex2_3, select_chain, kb)
    run("ex2.3", "returns 67 qubits", ff.grade_ex2_3, lambda b, n_qubits=68: select_chain(b, n_qubits)[:-1], kb)
    # reported only: under the expected-error cost (74 CZ/bond + 150 sx/qubit + readout) the O3 transpiler layout on
    # FakeKingston is itself within ~2 % of the baseline; hard-coding it is caught by the hidden-backend variant below
    run("ex2.3", "reported: transpiler O3 layout as chain", ff.grade_ex2_3, lambda b, n_qubits=68: list(layout), kb)
    run("ex2.3", "hard-coded Kingston chain, hidden FakeFez", ff.grade_ex2_3, select_chain_hardcoded_kingston, FakeFez())
    run("ex2.3", "perfect, hidden FakeFez", ff.grade_ex2_3, select_chain, FakeFez())
    plan = {"backend": "ibm_kingston", "chain": list(layout), "num_randomizations": 64, "shots_per_randomization": 256, "dd_sequence": "XY4",
            "predicted_usage_s": 2 + 0.45e-3 * 4 * 64 * 256, "predicted_sigma_X": 0.03}
    opts = EstimatorOptions(resilience_level=0, max_execution_time=150)
    opts.twirling.enable_gates = True; opts.twirling.enable_measure = True; opts.twirling.num_randomizations = 64; opts.twirling.shots_per_randomization = 256
    opts.dynamical_decoupling.enable = True; opts.dynamical_decoupling.sequence_type = "XY4"
    run("ex2.4", "perfect", ff.grade_ex2_4, plan, opts, circuits_all_isa)
    run("ex2.4", "dict options, resilience 2, too many shots", ff.grade_ex2_4, dict(plan, num_randomizations=200, shots_per_randomization=1000, predicted_usage_s=30),
        {"resilience_level": 2, "twirling": {"enable_gates": True, "enable_measure": True}, "dynamical_decoupling": {"enable": True}, "max_execution_time": 300}, circuits_all_isa)

    # ---- Part 3
    run("ex3.1", "perfect", ff.grade_ex3_1, R.odr_mitigate, odr_uncertainty, odr_bias)
    run("ex3.1", "ODR without post-selection", ff.grade_ex3_1, odr_no_postselection, odr_uncertainty, odr_bias)
    run("ex3.1", "bias with wrong ratio", ff.grade_ex3_1, R.odr_mitigate, odr_uncertainty, lambda c, fp, fc: (1 - np.asarray(c)) * (1 - np.asarray(fc) / np.asarray(fp)))
    run("ex3.2", "perfect", ff.grade_ex3_2, mitigation_options)
    run("ex3.2", "mutates the base options", ff.grade_ex3_2, lambda *a, **k: mitigation_options(*a, mutate=True, **k))
    run("ex3.2", "no measurement twirling, resilience_level 2", ff.grade_ex3_2, lambda *a, **k: mitigation_options(*a, measure_twirl=False, level=2, **k))
    run("ex3.2", "max_execution_time raised to 600", ff.grade_ex3_2, lambda *a, **k: mitigation_options(*a, keep_met=False, **k))
    run("ex3.2", "TREX always on", ff.grade_ex3_2, lambda b, n, s, dd_sequence="XY4", readout_mitigation=False: mitigation_options(b, n, s, dd_sequence, True))
    run("ex3.2", "returns None", ff.grade_ex3_2, lambda *a, **k: None)

    # ex 3.3: noisy rehearsal (cached)
    L6 = 6; t4 = 4.0; shots = 4000
    cache = os.path.join(ROOT, "scratch", f"ex33_cache_{shots}.npz")
    qw6 = R.prep_wave(L6); qv6 = R.prep_vacuum_for_subtraction(L6)
    qpw6, qmw6 = R.evolve_circuits(qw6, L6, t4, protect_midpoint=True); qpv6, qmv6 = R.evolve_circuits(qv6, L6, t4, protect_midpoint=True)
    isa0 = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42).run(qpw6)
    chain6 = isa0.layout.initial_index_layout(filter_ancillas=True)
    nm = reduced_noise_model(kb, chain6)
    obs6 = R.chiral_condensate_observables(L6)
    cw_ex = np.array([np.real(Statevector(qw6).expectation_value(o)) for o in obs6]); cv_ex = np.array([np.real(Statevector(qv6).expectation_value(o)) for o in obs6])
    if os.path.exists(cache) and not NO_CACHE:
        d = np.load(cache); A = {k: d[k] for k in d.files}; sim_s = float(A.pop("seconds"))
    else:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
        pm6 = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42, initial_layout=chain6)
        isas = [pm6.run(c) for c in (qpw6, qmw6, qpv6, qmv6)]
        est = AerEstimatorV2(options={"backend_options": {"noise_model": nm, "method": "statevector"}, "run_options": {"shots": shots, "seed_simulator": 11}})
        t0 = time.time(); res = est.run([(c, [o.apply_layout(c.layout) for o in obs6]) for c in isas]).result(); sim_s = time.time() - t0
        A = {k: r.data.evs for k, r in zip(("chi_wave", "chi_wave_mitig", "chi_vacuum", "chi_vacuum_mitig"), res)}
        np.savez(cache, **A, seconds=sim_s)
    A["chi_wave_exact"] = cw_ex; A["chi_vacuum_exact"] = cv_ex
    X_raw = A["chi_wave"] - A["chi_vacuum"]
    X_mit = R.odr_mitigate(A["chi_wave"], A["chi_wave_mitig"], cw_ex, L6) - R.odr_mitigate(A["chi_vacuum"], A["chi_vacuum_mitig"], cv_ex, L6)
    run("ex3.3", f"perfect (Aer statevector, 4x{shots} shots, {sim_s:.0f}s sim)", ff.grade_ex3_3, nm, chain6, A, X_raw, X_mit, R.odr_mitigate, kb)
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm_wrong = NoiseModel(basis_gates=["cz", "sx", "x", "rz", "id"]); nm_wrong.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), "cz")
    run("ex3.3", "generic noise model, X_mit inconsistent", ff.grade_ex3_3, nm_wrong, chain6, A, X_raw, X_raw, R.odr_mitigate, kb)

    # ---- Part 4 (cached hardware data)
    canary = json.load(open(os.path.join(ROOT, "organizer", "fallback_data", "canary_ibm_boston_2026-07-27.json")))
    sgn = np.array([(-1) ** j for j in range(68)])
    cw = sgn * np.array(canary["measured"]["t0_wave"]) + 1; cv = sgn * np.array(canary["measured"]["t0_vacuum"]) + 1
    ew, evac = ff._qdc_t0_exact()
    ret = (1 - cv) / (1 - evac)
    flagged = [int(j) for j in np.where(ret < 0.4)[0]]
    run("ex4.1", "ibm_boston canary 2026-07-27 (20000 shots)", ff.grade_ex4_1, {"chi_wave": cw, "chi_vacuum": cv, "flagged_sites": flagged, "verdict": canary["verdict"]},
        {"job_id": canary["job_id"], "backend": "ibm_boston", "usage_s": 18.0, "layout": canary["layout"]})
    hw = np.load(os.path.join(ROOT, "reference_data", "hardware_ibm_kingston_2026-07-25.npz"))

    def hw_result(strategy):
        d = {}
        for k, key in (("chi_wave", "z_wave"), ("chi_wave_mitig", "z_mitig_wave"), ("chi_vacuum", "z_vacuum"), ("chi_vacuum_mitig", "z_mitig_vacuum")):
            d[k] = sgn * hw[f"{strategy}__T8__{key}"] + 1; d[k + "_std"] = hw[f"{strategy}__T8__{key}_std"]
        d["X_raw"] = d["chi_wave"] - d["chi_vacuum"]
        d["X_mit"] = R.odr_mitigate(d["chi_wave"], d["chi_wave_mitig"], ew, 34) - R.odr_mitigate(d["chi_vacuum"], d["chi_vacuum_mitig"], evac, 34)
        d["sigma"] = odr_uncertainty(d["chi_wave"], d["chi_wave_std"], d["chi_wave_mitig"], d["chi_wave_mitig_std"], ew, 34, n_samples=400, seed=1)
        d["sigma"] = np.sqrt(d["sigma"] ** 2 + odr_uncertainty(d["chi_vacuum"], d["chi_vacuum_std"], d["chi_vacuum_mitig"], d["chi_vacuum_mitig_std"], evac, 34, n_samples=400, seed=2) ** 2)
        return d
    job = {"job_id": "d9hr3p50k0jc738il8dg", "backend": "ibm_kingston", "usage_s": 96.0, "layout": list(hw["initial_layout"]), "num_randomizations": 100, "shots_per_randomization": 1000}
    odr = hw_result("odr")
    run("ex4.2", "cached Kingston ODR set (T=8)", ff.grade_ex4_2, odr, job, R.odr_mitigate)
    run("ex4.2", "same, marked fallback", ff.grade_ex4_2, dict(odr, fallback=True), job, R.odr_mitigate)
    run("ex4.2", "fallback, organizer usage estimate 182 s", ff.grade_ex4_2, dict(odr, fallback=True), dict(job, usage_s=182.0, fallback=True), R.odr_mitigate)
    run("ex4.2", "real job, usage 182 s (-5)", ff.grade_ex4_2, odr, dict(job, usage_s=182.0), R.odr_mitigate)
    run("ex4.2", "X_mit tampered (+0.1 in the window)", ff.grade_ex4_2, dict(odr, X_mit=odr["X_mit"] + 0.1 * (np.arange(68) >= 25) * (np.arange(68) <= 42)), job, R.odr_mitigate)
    raw = hw_result("twirl_dd")
    run("ex4.2", "raw twirl+DD set (T=8)", ff.grade_ex4_2, raw, job, R.odr_mitigate)
    # z-score of the RMSE_W change, as the grader computes it (linear propagation)
    def rmse_stats(d):
        res = ff.score_hardware(d, job, R.odr_mitigate); W = np.arange(25, 43); Xref = ff.x_ref_t8(64)
        xw = res["X_hat"][W]; m = np.isfinite(xw); n = m.sum(); rm = res["rmse_w"]
        grad = (xw[m] - Xref[W][m]) / (n * rm); return rm, float(np.sqrt(np.sum((grad * np.asarray(res["sigma"])[W][m]) ** 2)))
    (rb, sb), (ri, si) = rmse_stats(raw), rmse_stats(odr)
    z = (rb - ri) / np.sqrt(sb**2 + si**2)
    rationale = ("Baseline: the twirl+DD run (no ODR calibration circuits). Improvement: the same layout with the forward-backward "
                 "mitigation circuits added so that operator decoherence renormalization can rescale each site; the z-score compares "
                 "RMSE_W of the two vacuum-subtracted profiles using the Monte-Carlo propagated per-site sigmas.")
    run("ex4.3", "Kingston twirl_dd -> odr", ff.grade_ex4_3, {"baseline": raw, "improved": odr, "usage_s": 45.0, "rationale": rationale, "z_score": float(z)}, R.odr_mitigate)
    run("ex4.3", "same as illustration on fallback data, 182 s", ff.grade_ex4_3, {"baseline": dict(raw, fallback=True), "improved": dict(odr, fallback=True), "usage_s": 182.0, "rationale": rationale, "z_score": float(z), "illustration": True}, R.odr_mitigate)
    run("ex4.3", "real run, usage 182 s", ff.grade_ex4_3, {"baseline": raw, "improved": odr, "usage_s": 182.0, "rationale": rationale, "z_score": float(z)}, R.odr_mitigate)

    # ---- Bonus
    q16, _ = R.evolve_circuits(qw, L, 8.0, n_steps=16); q32, _ = R.evolve_circuits(qw, L, 8.0, n_steps=32)
    Xref = ff.x_ref_t8(64); X05 = Xref + 0.01; X025 = Xref + 0.0025
    run("B1", "perfect (structural)", ff.grade_bonus_B1, q16, q32, X05, X025, (4 * X025 - X05) / 3)
    run("B1", "8-step circuit passed as 16", ff.grade_bonus_B1, qpw, q32, X05, X025, (4 * X025 - X05) / 3)
    run("B2", "perfect ordering", ff.grade_bonus_B2, {"raw_depol": 0.12, "odr_depol": 0.004, "raw_amp": 0.10, "odr_amp": 0.03, "witness_depol": 0.03, "witness_amp": 0.16})
    run("B2", "wrong ordering", ff.grade_bonus_B2, {"raw_depol": 0.12, "odr_depol": 0.05, "raw_amp": 0.10, "odr_amp": 0.03, "witness_depol": 0.03, "witness_amp": 0.02})
    run("B3", "perfect (rzz barbell)", ff.grade_bonus_B3, barbell_rzz)
    run("B3", "cx barbell", ff.grade_bonus_B3, lambda *a: cu.barbell(*a))

    # ---- regression tests for the 2026-09-10 review fixes -------------------------------
    # finding: Pauli.__str__ truncates at 50 qubits, so a sign flip on a high-index site used to pass
    logical = ff._chi_observables(34)
    obs_flip = [o.copy() for o in logical]
    bad_isa = []
    for c in circuits_all_isa:
        row = []
        for j, o in enumerate(logical):
            oo = o.apply_layout(c.layout)
            if j >= 20:                       # sites whose physical qubit sits beyond label position 50
                oo = SparsePauliOp(oo.paulis, -oo.coeffs)
            row.append(oo)
        bad_isa.append(row)
    run("ex2.1", "observables sign-flipped on high sites", ff.grade_ex2_1, circuits_all_isa, bad_isa, kb)

    # finding: graders must score 0 with a hint, never raise.  Rows marked "reported:" are inputs
    # that are ODD but not wrong (measurements on a prep circuit, 1-element lists, a missing wall
    # time): the grader must accept them, so full marks there is the expected outcome.
    crashers = [
        ("ex0.3", "reported: prep_vacuum with measure_all", ff.grade_ex0_3, (lambda L, a, b: (lambda q: (q.measure_all(), q)[1])(R.prep_vacuum(L, a, b)),)),
        ("ex0.5", "trotter_step ignores L", ff.grade_ex0_5, (lambda qc, L, dt, m, g: R.trotter_step(QuantumCircuit(16), 8, dt, m, g), R.evolve_circuits, R.prep_wave)),
        ("ex1.3", "reported: vqe values as 1-element lists", ff.grade_ex1_3, ({L: [float(REFS[f"vqe_fid_L{L}"])] for L in (4, 6, 8)},
                                                                   {L: [float(REFS[f"E_adapt_L{L}"] - REFS[f"E0_L{L}"])] for L in (4, 6, 8)})),
        ("ex1.5", "reported: None wall time (not graded)", ff.grade_ex1_5, ({8: 0.05, 20: 0.01, 40: 0.001}, {20: None}, ff.x_ref_t8(64))),
    ]
    for ex, var, fn, args in crashers:
        try:
            run(ex, var, fn, *args)
        except Exception as e:  # noqa: BLE001
            TABLE.append((ex, var, -1.0, ff.MAX_POINTS[ex], f"RAISED {type(e).__name__}: {e}"))

    # finding: numpy scalars / size-1 arrays must be accepted
    run("ex1.1", "reported: E0_L8 as a (1,) array", ff.grade_ex1_1, R.schwinger_hamiltonian, R.electric_hamiltonian_truncated,
        np.array([float(REFS["E0_L8"])]))
    run("ex1.4", "reported: string table keys + 0-d richardson", ff.grade_ex1_4,
        {f"t={t:g}, dt={dt:g}": float(REFS[f"trotter_table_t{t}_dt{dt:g}"]) for t in (2, 4) for dt in (1, 0.5, 0.25)},
        np.array(float(REFS["richardson_error_t4"])), "trotter")

    # finding: an "improvement" without uncertainties must not collect the z >= 2 point
    nosig = {k: (v if not (isinstance(k, str) and k.endswith("_std")) else None) for k, v in odr.items()}
    nosig = {k: v for k, v in nosig.items() if v is not None and not (isinstance(k, str) and k.endswith("_std"))}
    nosig_raw = {k: v for k, v in raw.items() if not (isinstance(k, str) and k.endswith("_std"))}
    nosig.pop("sigma", None); nosig_raw.pop("sigma", None)
    run("ex4.3", "improvement with no uncertainties (z undefined)", ff.grade_ex4_3,
        {"baseline": nosig_raw, "improved": nosig, "usage_s": 40.0, "rationale": rationale, "z_score": 5.0}, R.odr_mitigate)

    # finding: NaN-ing window sites must not buy a better rank
    met_full = ff.hardware_metric(odr["X_mit"], np.full(68, 0.02))
    X_holes = odr["X_mit"].copy(); worst = np.argsort(-np.abs(odr["X_mit"][25:43] - ff.x_ref_t8(64)[25:43]))[:2] + 25
    X_holes[worst] = np.nan
    met_holes = ff.hardware_metric(X_holes, np.full(68, 0.02))
    TABLE.append(("ex4.2", "reported: NaN-ing 2 worst sites", met_holes["points_raw"], 18,
                  f"points {met_full['points_raw']:.1f} -> {met_holes['points_raw']:.1f}; RANK-RMSE {met_full['rmse_w_rank']:.3f} -> {met_holes['rmse_w_rank']:.3f} (must not improve)"))
    assert met_holes["rmse_w_rank"] >= met_full["rmse_w_rank"] - 1e-12, "NaN-ing sites improved the rank metric"
    assert met_holes["points_raw"] <= met_full["points_raw"] + 1e-9, "NaN-ing sites improved the points"
    # ... and no SINGLE window site either (each hidden site is charged NAN_SITE_ERROR), nor the quiet test
    Xr = ff.x_ref_t8(64); gains = []; skipped = []
    for j in range(25, 43):
        Xh = odr["X_mit"].copy(); Xh[j] = np.nan
        mh = ff.hardware_metric(Xh, np.full(68, 0.02))
        if abs(odr["X_mit"][j] - Xr[j]) > max(abs(Xr[j]), ff.NAN_SITE_ERROR):
            skipped.append(j); continue
        if mh["points_raw"] > met_full["points_raw"] + 1e-9 or mh["rmse_w_rank"] < met_full["rmse_w_rank"] - 1e-12:
            gains.append(j)
    assert not gains, f"NaN-ing window site(s) {gains} improved points or rank"
    Xo = odr["X_mit"].copy(); loud = [j for j in np.argsort(-np.abs(odr["X_mit"])) if j < 25 or j > 42][:20]
    Xo[loud] = np.nan
    assert ff.hardware_metric(Xo, np.full(68, 0.02))["quiet"] >= met_full["quiet"] - 1e-12, "NaN-ing outside sites lowered quiet"
    TABLE.append(("ex4.2", "reported: NaN-ing any single window site", met_full["points_raw"], 18,
                  f"no single-site NaN gains points or rank (charged max(|X_ref|, {ff.NAN_SITE_ERROR})); sites with |err| above that: {skipped}"))

    # finding: plausible wrong RETURN VALUES must score 0 with a hint, never raise
    from qiskit.circuit import Parameter
    def prep_vacuum_unbound(L, a, b):
        q = R.prep_vacuum(L, a, b); q.rx(Parameter("t"), 0); return q
    run("ex0.3", "unbound Parameter in the circuit", ff.grade_ex0_3, prep_vacuum_unbound)
    run("ex1.2", "reported: electric_layer with measure_all", ff.grade_ex1_2,
        lambda L, t, g: (lambda q: (q.measure_all(), q)[1])(electric_layer(L, t, g)), REFS["truncation_shift_L8_t4"])
    run("ex3.1", "odr_mitigate returns a dict", ff.grade_ex3_1, lambda *a, **k: {"x": 1}, odr_uncertainty, odr_bias)
    run("ex3.1", "odr_bias returns a string", ff.grade_ex3_1, R.odr_mitigate, odr_uncertainty, lambda *a, **k: "big")
    run("ex3.3", "chain is None", ff.grade_ex3_3, nm, None, A, X_raw, X_mit, R.odr_mitigate, kb)
    run("ex3.3", "X_raw has 68 values", ff.grade_ex3_3, nm, chain6, A, np.zeros(68), X_mit, R.odr_mitigate, kb)
    run("ex3.3", "odr_mitigate returns 68 values", ff.grade_ex3_3, nm, chain6, A, X_raw, X_mit, lambda *a, **k: np.zeros(68), kb)
    run("ex4.2", "odr_mitigate returns 69 values", ff.grade_ex4_2, odr, job, lambda *a, **k: np.zeros(69))
    run("ex4.2", "hardware_result without X_mit", ff.grade_ex4_2, {k: v for k, v in odr.items() if k != "X_mit"}, job, R.odr_mitigate)
    run("B2", "toy value None", ff.grade_bonus_B2, {"raw_depol": 0.1, "odr_depol": None, "raw_amp": 0.1, "odr_amp": 0.05, "witness_depol": 0.03, "witness_amp": 0.16})
    run("ex2.4", "reported: options with numpy bools", ff.grade_ex2_4, plan, {"twirling": {"enable_gates": np.True_, "enable_measure": np.True_},
        "dynamical_decoupling": {"enable": np.True_}, "resilience_level": np.int64(0), "max_execution_time": np.int64(180)}, None)
    run("ex4.1", "reported: exact t=0 profiles as canary (judge flag)", ff.grade_ex4_1,
        {"chi_wave": ew, "chi_vacuum": evac, "flagged_sites": [], "verdict": "PASS"},
        {"job_id": "a" * 20, "backend": "ibm_kingston", "usage_s": 4.0, "layout": list(range(68))})

    # finding: the notebook's no-data stub (no job, no fallback file) must score 0, not the free metadata point
    stub_ji = {"job_id": "none", "backend": "ibm_kingston", "usage_s": 0.0, "num_randomizations": 0, "shots_per_randomization": 0,
               "fallback": True, "no_data": True, "layout": list(range(68))}
    p41 = run("ex4.1", "no-data stub (job 'none', all-NaN evs)", ff.grade_ex4_1,
              {"chi_wave": np.full(68, np.nan), "chi_vacuum": np.full(68, np.nan), "evs": np.full((2, 68), np.nan),
               "flagged_sites": [], "verdict": "FAIL", "fallback": True}, stub_ji)
    assert p41 == 0.0, "no-data canary stub must score 0"
    p42 = run("ex4.2", "no-data stub (all-NaN evs)", ff.grade_ex4_2,
              {**{k: np.full(68, np.nan) for k in ("chi_wave", "chi_wave_mitig", "chi_vacuum", "chi_vacuum_mitig")},
               "evs": np.full((4, 68), np.nan), "stds": np.zeros((4, 68)), "X_mit": np.full(68, np.nan), "sigma": np.zeros(68), "fallback": True},
              stub_ji, R.odr_mitigate)
    assert p42 == 0.0, "no-data hardware stub must score 0"

    # ---- table
    print("\n== results")
    print(f"{'exercise':7s} {'variant':45s} {'pts':>9s}  verdict")
    fails = []
    for ex, var, pts, mx, det in TABLE:
        print(f"{ex:7s} {var[:45]:45s} {pts:4.1f}/{mx:<3d}  {det}")
        if var.startswith("perfect") and pts < mx - 1e-9 and not ex.startswith("ex4"):
            fails.append((ex, var, pts, mx))
        if not var.startswith(("perfect", "reported")) and not ex.startswith("ex4") and pts >= mx - 1e-9:
            fails.append((ex, var, pts, mx))
        if pts < 0:                      # the grader raised instead of scoring 0 with a hint
            fails.append((ex, var, pts, mx))
    print("\n== summary()"); ff.summary()
    print(f"\ntotal test time {time.time() - T_START:.0f}s")
    if fails:
        print("UNEXPECTED:", fails); sys.exit(1)
    print("ALL GRADER TESTS PASSED")


if __name__ == "__main__":
    main()
