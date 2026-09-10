"""(c) plausible input types / shapes that could crash a grader (contract: raise only on wrong types, never on a wrong answer)."""
from review_grader_common import setup, run, ROOT
import os, sys, json, numpy as np
ff = setup("crash")
import schwinger_reference as R
sys.path.insert(0, os.path.join(ROOT, "tools")); import test_grader as T
ff.SUBMISSION_DIR = os.path.join(ROOT, "scratch", "review_sub_crash")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeKingston
from qiskit.primitives.containers import ObservablesArray
r = ff._refs(); kb = FakeKingston()

# ex0.5: trotter_step that ignores L (always builds an L=8 circuit) -- the anti-hard-coding case

# --- rebuild the ISA circuits (as in review_grader_crash.py)
L = 34
qw = R.prep_wave(L); qv = R.prep_vacuum_for_subtraction(L)
qpw, qmw = R.evolve_circuits(qw, L, 8.0, protect_midpoint=True); qpv, qmv = R.evolve_circuits(qv, L, 8.0, protect_midpoint=True)
from qiskit.transpiler import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42)
isa_w = pm.run(qpw); layout = isa_w.layout.initial_index_layout(filter_ancillas=True)
pm_l = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42, initial_layout=layout)
circs = [isa_w] + [pm_l.run(c) for c in (qmw, qpv, qmv)]
obs = [o.apply_layout(isa_w.layout) for o in R.chiral_condensate_observables(L)]
# why does an ObservablesArray fail at observable 15?
oa = ff._as_observable_list(ObservablesArray(obs))
for j in (0, 1, 15):
    print(f"obs[{j}] from ObservablesArray:", ff._pauli_dict(oa[j], False), " expected:", ff._pauli_dict(obs[j], False))
    print("   raw ObservablesArray entry:", ObservablesArray(obs).tolist()[j])
run("ex2.1 circuits as tuple", ff.grade_ex2_1, tuple(circs), obs, kb)
plan = {"backend": "ibm_kingston", "chain": np.asarray(layout), "num_randomizations": np.int64(64), "shots_per_randomization": 256, "dd_sequence": "XY4",
        "predicted_usage_s": np.float64(2 + 0.45e-3 * 4 * 64 * 256), "predicted_sigma_X": np.full(68, 0.03)}
from qiskit_ibm_runtime.options import EstimatorOptions
opts = EstimatorOptions(resilience_level=0, max_execution_time=150)
opts.twirling.enable_gates = True; opts.twirling.enable_measure = True; opts.dynamical_decoupling.enable = True
run("ex2.4 predicted_sigma_X as per-site array, numpy scalars", ff.grade_ex2_4, plan, opts, circs)
run("ex2.4 circuits_all_isa=None", ff.grade_ex2_4, dict(plan, predicted_sigma_X=0.03), opts, None)
run("ex2.4 options with max_execution_time unset", ff.grade_ex2_4, dict(plan, predicted_sigma_X=0.03), EstimatorOptions(resilience_level=0, twirling={"enable_gates": True, "enable_measure": True}, dynamical_decoupling={"enable": True}), circs)
# ex2.2 with a layout given as numpy array and evolve_circuits_matched ignoring L (returns 68-qubit circuits at L=6!)
n2q = int(r["n2q_logical_t8"]); ncz = int(r["n_cz_physics_t8_kingston_O3_seed42"])
run("ex2.2 n2q as np.int64, layout numpy", ff.grade_ex2_2, np.int64(n2q), np.int64(ncz), T.evolve_circuits_matched, R.prep_wave, kb, np.asarray(layout))
run("ex2.2 n_cz_physics as float", ff.grade_ex2_2, n2q, float(ncz), T.evolve_circuits_matched, R.prep_wave, kb, layout)
# ex3.3 with lists / numpy chain
d = np.load(os.path.join(ROOT, "scratch", "ex33_cache_4000.npz")); A = {k: d[k].tolist() for k in d.files if k != "seconds"}
chain6 = [52, 53, 54, 55, 59, 75, 74, 73, 79, 93, 94, 95]
nm = T.reduced_noise_model(kb, chain6)
cw_ex = r["chi_wave_t0_L6"]; cv_ex = r["chi_vacuum_sub_t0_L6"]
X_raw = np.array(A["chi_wave"]) - np.array(A["chi_vacuum"])
X_mit = R.odr_mitigate(A["chi_wave"], A["chi_wave_mitig"], cw_ex, 6) - R.odr_mitigate(A["chi_vacuum"], A["chi_vacuum_mitig"], cv_ex, 6)
run("ex3.3 lists + numpy chain + list X_raw", ff.grade_ex3_3, nm, np.asarray(chain6), A, X_raw.tolist(), X_mit, R.odr_mitigate, kb)
run("ex3.3 noise model built with add_all_qubit_quantum_error", ff.grade_ex3_3, __import__("qiskit_aer").noise.NoiseModel(), chain6, A, X_raw, X_mit, R.odr_mitigate, kb)
# ex4.1 odd job_info types
ew, evac = ff._qdc_t0_exact()
run("ex4.1 layout given as int / flagged None", ff.grade_ex4_1, {"chi_wave": ew, "chi_vacuum": evac, "flagged_sites": None, "verdict": "x"}, {"job_id": "x", "backend": "b", "usage_s": 3, "layout": 68})
run("ex4.1 arrays as lists, evs key", ff.grade_ex4_1, {"evs": [list(ew), list(evac)], "flagged_sites": [], "verdict": "x"}, {"job_id": "x", "backend": "b", "usage_s": 3.0, "layout": list(range(68))})
# ex4.2 hardware_result with stds=None, usage_s string
hw = np.load(os.path.join(ROOT, "reference_data", "hardware_ibm_kingston_2026-07-25.npz")); sgn = np.array([(-1) ** j for j in range(68)])
hr = {}
for k, key in (("chi_wave", "z_wave"), ("chi_wave_mitig", "z_mitig_wave"), ("chi_vacuum", "z_vacuum"), ("chi_vacuum_mitig", "z_mitig_vacuum")):
    hr[k] = list(sgn * hw[f"odr__T8__{key}"] + 1); hr[k + "_std"] = None
hr["X_mit"] = R.odr_mitigate(hr["chi_wave"], hr["chi_wave_mitig"], ew, 34) - R.odr_mitigate(hr["chi_vacuum"], hr["chi_vacuum_mitig"], evac, 34)
run("ex4.2 lists, stds None, no sigma, usage_s string", ff.grade_ex4_2, hr, {"job_id": "x", "backend": "ibm_kingston", "usage_s": "96", "layout": list(hw["initial_layout"])}, R.odr_mitigate)
run("ex4.2 evs as a (4,68) array under key 'evs' only", ff.grade_ex4_2, {"evs": np.array([hr["chi_wave"], hr["chi_wave_mitig"], hr["chi_vacuum"], hr["chi_vacuum_mitig"]]), "X_mit": hr["X_mit"]}, {"job_id": "x", "backend": "ibm_kingston", "usage_s": 96.0, "layout": list(hw["initial_layout"])}, R.odr_mitigate)
# ex4.3 improvement with z_score None / usage None
run("ex4.3 z_score None, usage None", ff.grade_ex4_3, {"baseline": hr, "improved": hr, "usage_s": None, "rationale": "y" * 200, "z_score": None}, R.odr_mitigate)
# B1 with circuits as lists / X as lists; B2 with numpy values; B3 returning a Gate
run("B2 values as 0-d arrays", ff.grade_bonus_B2, {k: np.asarray(v) for k, v in {"raw_depol": 0.12, "odr_depol": 0.004, "raw_amp": 0.10, "odr_amp": 0.03, "odr_amp_twirl": 0.006}.items()})
run("B3 returns a Gate (to_gate)", ff.grade_bonus_B3, lambda *a: T.barbell_rzz(*a).to_gate())
run("B3 returns an Operator", ff.grade_bonus_B3, lambda *a: __import__("qiskit").quantum_info.Operator(T.barbell_rzz(*a)))
