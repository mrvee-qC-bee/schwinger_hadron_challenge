"""(a) more WRONG solutions that should not pass, and metric-gaming checks."""
from review_grader_common import setup, run, ROOT
import os, json, numpy as np
ff = setup("wrong")
import schwinger_reference as R
import challenge_utils as cu
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKingston
r = ff._refs(); kb = FakeKingston()

# ex0.5: correct trotter_step, but evolve_circuits returns the INITIAL circuit as the mitigation circuit (no gates)
def evolve_identity_mitig(qc_init, L, t, m, g):
    qc, _ = R.evolve_circuits(qc_init, L, t, m, g)
    return qc, qc_init.copy()
run("ex0.5 mitigation circuit = qc_init (no gates)", ff.grade_ex0_5, R.trotter_step, evolve_identity_mitig, R.prep_wave)
# ex0.5: mitigation circuit = physics circuit followed by its inverse (returns, but 2x the depth: not the QDC circuit)
def evolve_inverse_mitig(qc_init, L, t, m, g):
    qc, _ = R.evolve_circuits(qc_init, L, t, m, g)
    qm = qc_init.copy(); ev = QuantumCircuit(2 * L); ev = R.evolve_circuits(ev, L, t, m, g)[0]
    qm.compose(ev, inplace=True); qm.compose(ev.inverse(), inplace=True)
    return qc, qm
run("ex0.5 mitigation circuit = U U^dagger (full forward + full inverse)", ff.grade_ex0_5, R.trotter_step, evolve_inverse_mitig, R.prep_wave)

# ex2.2: evolve_circuits_matched returns (physics, physics.copy()) -> CZ accounting trivially identical
layout = [int(q) for q in r["layout_t8_kingston_O3_seed42"]]
n2q = int(r["n2q_logical_t8"]); ncz = int(r["n_cz_physics_t8_kingston_O3_seed42"])
run("ex2.2 matched mitig := copy of the physics circuit", ff.grade_ex2_2, n2q, ncz, lambda qi, L, t, m, g: (R.evolve_circuits(qi, L, t, m, g)[0], R.evolve_circuits(qi, L, t, m, g)[0].copy()), R.prep_wave, kb, layout)

# ex2.3: chain reusing a dead edge / repeated qubit / not a path
summ = ff.target_summary(kb)
dead = [(int(a), int(b)) for (a, b), e in zip(summ["edges"], summ["cz_error"]) if e >= 0.5]
print("dead edges on frozen Kingston:", dead)
good = ff.baseline_chain(summ, 68)[1]
def with_dead(backend, n_qubits=68):
    # splice the dead edge into a chain: take a valid 66-chain and glue the dead pair at one end if adjacent
    a, b = dead[0]
    # find a chain that ends next to a: use the baseline and just append the dead pair (may not be adjacent -> also invalid)
    return good[:66] + [a, b]
run("ex2.3 chain ending with a dead edge (also a non-edge join)", ff.grade_ex2_3, with_dead, kb)
run("ex2.3 chain with a repeated qubit", ff.grade_ex2_3, lambda b, n_qubits=68: good[:67] + [good[0]], kb)
run("ex2.3 select_chain returns (chain, cost) tuple", ff.grade_ex2_3, lambda b, n_qubits=68: (good, 1.0), kb)
run("ex2.3 select_chain with signature (backend, n)", ff.grade_ex2_3, lambda backend, n=68: good, kb)

# ex3.2: postselect returning probabilities instead of counts
def postselect_probs(counts, L):
    keep = {k: v for k, v in counts.items() if k.count("1") == L}; tot = sum(keep.values())
    return {k: v / tot for k, v in keep.items()}

import sys; sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["FF_SUBMISSION_DIR"] = ff.SUBMISSION_DIR
import test_grader as T   # perfect-participant wrappers (import sets FF_SUBMISSION_DIR to scratch/submission_test; restore)
ff.SUBMISSION_DIR = os.path.join(ROOT, "scratch", "review_sub_wrong")
run("ex3.2 postselect returns probabilities", ff.grade_ex3_2, T.twirl_circuit, postselect_probs)

# ex4.2: ODR WITHOUT post-selection but with a consistent X_mit -> local 4.2 does not check the formula
ew, evac = ff._qdc_t0_exact(); hw = np.load(os.path.join(ROOT, "reference_data", "hardware_ibm_kingston_2026-07-25.npz"))
sgn = np.array([(-1) ** j for j in range(68)])
def hw_result(strategy, odr):
    d = {}
    for k, key in (("chi_wave", "z_wave"), ("chi_wave_mitig", "z_mitig_wave"), ("chi_vacuum", "z_vacuum"), ("chi_vacuum_mitig", "z_mitig_vacuum")):
        d[k] = sgn * hw[f"{strategy}__T8__{key}"] + 1; d[k + "_std"] = hw[f"{strategy}__T8__{key}_std"]
    d["X_mit"] = odr(d["chi_wave"], d["chi_wave_mitig"], ew, 34) - odr(d["chi_vacuum"], d["chi_vacuum_mitig"], evac, 34)
    d["sigma"] = np.full(68, 0.03)
    return d
job = {"job_id": "d9hr3p50k0jc738il8dg", "backend": "ibm_kingston", "usage_s": 96.0, "layout": list(hw["initial_layout"])}
odr_nops = T.odr_no_postselection
run("ex4.2 team ODR = per-site, no post-selection, consistent X_mit", ff.grade_ex4_2, hw_result("odr", odr_nops), job, odr_nops)
# a rigged odr_mitigate: behaves correctly on synthetic data, returns X_ref-derived values on the real t=0 references
Xref = ff.x_ref_t8(64)
def odr_rigged(chi, chi_cal, chi_exact, L, suppression_threshold=0.01):
    chi_exact = np.asarray(chi_exact, float)
    if L == 34 and np.allclose(chi_exact, ew):
        return ff._mps()["chi_wave_t8_bd64"] + 1e-3 * np.sin(np.arange(68))
    if L == 34 and np.allclose(chi_exact, evac):
        return ff._mps()["chi_vacuum_t8_bd64"]
    return R.odr_mitigate(chi, chi_cal, chi_exact, L, suppression_threshold)
run("ex3.1 rigged odr_mitigate (passes synthetic data)", ff.grade_ex3_1, odr_rigged, T.odr_uncertainty, T.odr_bias)
run("ex4.2 rigged odr_mitigate on the real Kingston evs", ff.grade_ex4_2, hw_result("odr", odr_rigged), job, odr_rigged)

# ex4.3: no sigma and no stds anywhere -> grader sigma = 0 -> z = inf
base = hw_result("twirl_dd", R.odr_mitigate); imp = hw_result("odr", R.odr_mitigate)
for d in (base, imp):
    for k in list(d):
        if k.endswith("_std") or k == "sigma":
            del d[k]
rat = "x" * 160
run("ex4.3 no sigma / no stds in either run (z should be undefined)", ff.grade_ex4_3, {"baseline": base, "improved": imp, "usage_s": 30.0, "rationale": rat, "z_score": 5.0}, R.odr_mitigate)
# same runs swapped: 'improvement' is actually worse
run("ex4.3 same, baseline/improved swapped", ff.grade_ex4_3, {"baseline": imp, "improved": base, "usage_s": 30.0, "rationale": rat, "z_score": 5.0}, R.odr_mitigate)

# ex4.2 metric gaming: NaN-ing the worst window sites (as if post-selected away)
odr = hw_result("odr", R.odr_mitigate)
X = odr["X_mit"].copy(); W = np.arange(25, 43)
m0 = ff.hardware_metric(X, np.full(68, 0.03))
print(f"metric on the Kingston ODR set: RMSE_W {m0['rmse_w']:.3f} points_raw {m0['points_raw']:.2f}")
order = W[np.argsort(-np.abs(X[W] - Xref[W]))]
for k in (1, 2, 3, 4, 6, 8):
    Xk = X.copy(); Xk[order[:k]] = np.nan
    mk = ff.hardware_metric(Xk, np.full(68, 0.03))
    print(f"  NaN the {k} worst window sites: RMSE_W {mk['rmse_w']:.3f}, p_rmse {mk['p_rmse']:.2f}, points_raw {mk['points_raw']:.2f} (delta {mk['points_raw']-m0['points_raw']:+.2f})")

# ex4.1 usage_s as numpy int / flagged as numpy array
canary = json.load(open(os.path.join(ROOT, "organizer", "fallback_data", "canary_ibm_boston_2026-07-27.json")))
cw = sgn * np.array(canary["measured"]["t0_wave"]) + 1; cv = sgn * np.array(canary["measured"]["t0_vacuum"]) + 1
ret = (1 - cv) / (1 - evac); flagged = np.where(ret < 0.4)[0]
run("ex4.1 usage_s = np.int64(18)", ff.grade_ex4_1, {"chi_wave": cw, "chi_vacuum": cv, "flagged_sites": flagged, "verdict": "fail"},
    {"job_id": canary["job_id"], "backend": "ibm_boston", "usage_s": np.int64(18), "layout": canary["layout"]})
