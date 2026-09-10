"""Determinism + runtime: perfect participant (test_grader wrappers) graded twice into two folders; compare score.json."""
from review_grader_common import setup, run, ROOT
import os, sys, json, time, numpy as np
ff = setup("det1")
import schwinger_reference as R
sys.path.insert(0, os.path.join(ROOT, "tools")); import test_grader as T
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez
from qiskit_ibm_runtime.options import EstimatorOptions
r = ff._refs(); kb = FakeKingston()
L = 34
qw = R.prep_wave(L); qv = R.prep_vacuum_for_subtraction(L)
qpw, qmw = R.evolve_circuits(qw, L, 8.0, protect_midpoint=True); qpv, qmv = R.evolve_circuits(qv, L, 8.0, protect_midpoint=True)
pm = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42)
isa_w = pm.run(qpw); layout = isa_w.layout.initial_index_layout(filter_ancillas=True)
pm_l = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42, initial_layout=layout)
circs = [isa_w] + [pm_l.run(c) for c in (qmw, qpv, qmv)]
obs = [o.apply_layout(isa_w.layout) for o in R.chiral_condensate_observables(L)]
n2q = sum(1 for i in qpw.decompose(reps=3).data if len(i.qubits) > 1); ncz = isa_w.count_ops().get("cz", 0)
rd = os.path.join(ROOT, "reference_data")
X40 = np.loadtxt(f"{rd}/chi_wave_evolved_sim_L34_maxbond40.txt") - np.loadtxt(f"{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
plan = {"backend": "ibm_kingston", "chain": list(layout), "num_randomizations": 64, "shots_per_randomization": 256, "dd_sequence": "XY4",
        "predicted_usage_s": 2 + 0.45e-3 * 4 * 64 * 256, "predicted_sigma_X": 0.03}
opts = EstimatorOptions(resilience_level=0, max_execution_time=150)
opts.twirling.enable_gates = True; opts.twirling.enable_measure = True; opts.dynamical_decoupling.enable = True
tab = {(t, dt): float(r[f"trotter_table_t{t}_dt{dt:g}"]) for t in (2, 4) for dt in (1.0, 0.5, 0.25)}
fid = {L_: float(r[f"vqe_fid_L{L_}"]) for L_ in (4, 6, 8)}; gap = {L_: float(r[f"E_adapt_L{L_}"] - r[f"E0_L{L_}"]) for L_ in (4, 6, 8)}

def full_pass(tag):
    sub = os.path.join(ROOT, "scratch", f"review_sub_{tag}"); os.makedirs(sub, exist_ok=True)
    for f in os.listdir(sub): os.remove(os.path.join(sub, f))
    ff.SUBMISSION_DIR = sub; ff._BASELINE_CACHE.clear()
    t0 = time.time(); times = {}
    def tr(ex, *a, **k):
        t1 = time.time(); run(f"[{tag}] {ex}", getattr(ff, "grade_" + ex.replace(".", "_")), *a, **k); times[ex] = round(time.time() - t1, 1)
    tr("ex0.1", R.chiral_condensate_observables); tr("ex0.2", R.RXYplus, R.RXYminus); tr("ex0.3", R.prep_vacuum); tr("ex0.4", R.prep_wave)
    tr("ex0.5", R.trotter_step, R.evolve_circuits, R.prep_wave)
    tr("ex1.1", R.schwinger_hamiltonian, R.electric_hamiltonian_truncated, float(r["E0_L8"])); tr("ex1.2", T.electric_layer, float(r["truncation_shift_L8_t4"]))
    tr("ex1.3", fid, gap); tr("ex1.4", tab, float(r["richardson_error_t4"]), "trotter"); tr("ex1.5", {8: 0.0561, 20: 0.0029, 40: 0.0007}, {8: 1.3, 20: 7.0, 40: 33.6}, X40)
    tr("ex2.1", circs, obs, kb); tr("ex2.2", n2q, ncz, T.evolve_circuits_matched, R.prep_wave, kb, layout)
    tr("ex2.3", T.select_chain, kb); tr("ex2.4", plan, opts, circs)
    tr("ex3.1", R.odr_mitigate, T.odr_uncertainty, T.odr_bias); tr("ex3.2", T.twirl_circuit, T.postselect_charge)
    print(f"[{tag}] total {time.time() - t0:.0f} s; per exercise {times}")
    return json.load(open(os.path.join(sub, "score.json")))

s1 = full_pass("det1"); s2 = full_pass("det2")
diff = []
for ex in s1:
    a = {k: v for k, v in s1[ex].items() if k not in ("time", "seconds")}; b = {k: v for k, v in s2[ex].items() if k not in ("time", "seconds")}
    if a != b:
        diff.append((ex, a, b))
print("\nDIFFERENCES between the two passes:", diff if diff else "none")
print("select_chain seconds pass1/pass2:", s1["ex2.3"].get("seconds"), s2["ex2.3"].get("seconds"))
# baseline timing on all three fake backends
for B in (FakeKingston, FakeFez):
    t0 = time.time(); cb, ch = ff.baseline_chain(ff.target_summary(B()), 68); print(f"baseline_chain({B.__name__}) {time.time()-t0:.2f} s cost {cb:.4f}")
