"""
Smoke test for tools/nb_part_b.py: executes every code cell of the second notebook half sequentially
with a STUB fallfest_grader, after pre-defining the Part 0-2 names from organizer/schwinger_reference.py.

    MPLBACKEND=Agg python tools/smoke_part_b.py            # full run (~5-8 min)
    MPLBACKEND=Agg python tools/smoke_part_b.py --view     # only write the participant view + leak check

Reports per-cell runtime, checks that the participant view (nbkit.strip_answers) leaks no solution
line, and writes scratch/part_b_participant_view.py for manual inspection.
"""
from __future__ import annotations

import os
import sys
import time
import types
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "organizer")]
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(os.path.join(ROOT, "scratch"), exist_ok=True)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nbkit  # noqa: E402
import nb_part_b  # noqa: E402

# --------------------------------------------------------------------------- stub grader
stub = types.ModuleType("fallfest_grader")
_calls: list[tuple[str, int]] = []


def _make(name):
    def grade(*args, **kwargs):
        _calls.append((name, len(args)))
        print(f"[stub {name}] called with {len(args)} positional args: {[type(a).__name__ for a in args]}")
        return 0.0
    return grade


for _n in ["check_env", "grade_ex0_1", "grade_ex0_2", "grade_ex0_3", "grade_ex0_4", "grade_ex0_5",
           "grade_ex1_1", "grade_ex1_2", "grade_ex1_3", "grade_ex1_4", "grade_ex1_5",
           "grade_ex2_1", "grade_ex2_2", "grade_ex2_3", "grade_ex2_4",
           "grade_ex3_1", "grade_ex3_2", "grade_ex3_3", "grade_ex4_1", "grade_ex4_2", "grade_ex4_3",
           "grade_bonus_B1", "grade_bonus_B2", "grade_bonus_B3"]:
    setattr(stub, _n, _make(_n))
stub.summary = lambda: print("[stub summary]", _calls)
if "--stub" in sys.argv or not os.path.exists(os.path.join(ROOT, "fallfest_grader.py")):
    sys.modules["fallfest_grader"] = stub
    print("using the STUB grader")
else:
    print("using the REAL fallfest_grader.py")


# --------------------------------------------------------------------------- participant-view leak check
def leak_check() -> None:
    n_code = 0
    leaked = []
    view_lines = []
    for i, c in enumerate(nb_part_b.CELLS):
        if c["cell_type"] != "code":
            continue
        n_code += 1
        src = c["source"]
        part = nbkit.strip_answers(src)
        if "solution-only" in c.get("tags", []):
            view_lines.append(f"# ===== cell {i}: SOLUTION-ONLY (dropped from participant build) =====\n")
            continue
        view_lines.append(f"# ===== cell {i} =====\n{part}\n")
        inside = False
        for line in src.splitlines():
            if nbkit._BEGIN.match(line):
                inside = True
                continue
            if nbkit._END.match(line):
                inside = False
                continue
            if inside and not nbkit._KEEP.search(line) and line.strip():
                if line in part.splitlines():
                    leaked.append((i, line))
            if nbkit._SOL.match(line) and line in part.splitlines():
                leaked.append((i, line))
        assert "# SOL" not in part and "# PARTICIPANT:" not in part
    with open(os.path.join(ROOT, "scratch", "part_b_participant_view.py"), "w") as fh:
        fh.write("".join(view_lines))
    print(f"participant view: {n_code} code cells, {len(leaked)} leaked solution lines")
    for i, line in leaked:
        print("   LEAK cell", i, ":", line)
    assert not leaked, "solution lines leaked into the participant view"
    # tokens that must never appear in the participant view
    text = "".join(view_lines)
    for tok in ('("resilience", "pec_mitigation"): False', "np.where(keep_a", "rng.normal(chi, s_phys", "depolarizing_error(eps_cz, 2)",
                "return (4.0 * np.asarray(X_half", "qc.rzz(theta, j, k)"):
        assert tok not in text, f"solution token leaked: {tok}"
    print("no solution tokens in the participant view")


# --------------------------------------------------------------------------- Part 0-2 namespace
def build_namespace() -> dict:
    import schwinger_reference as R
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    from qiskit.transpiler import generate_preset_pass_manager

    t0 = time.time()
    ns: dict = {}
    L, m, g, t = 34, 0.5, 0.3, 8.0
    ns.update(L=L, m=m, g=g, t=t)
    ns["chiral_condensate_observables"] = R.chiral_condensate_observables
    ns["prep_vacuum"] = R.prep_vacuum
    ns["prep_wave"] = R.prep_wave
    ns["trotter_step"] = R.trotter_step
    ns["evolve_circuits"] = R.evolve_circuits
    ns["observables"] = R.chiral_condensate_observables(L)
    ns["qc_vacuum_init"] = R.prep_vacuum_for_subtraction(L)
    ns["qc_wave_init"] = R.prep_wave(L)
    ns["chi_wave_exact"] = np.loadtxt("reference_data/chi_wave_t0_sim_L34.txt")      # == Aer MPS t=0 to 1e-11
    ns["chi_vacuum_exact"] = np.loadtxt("reference_data/chi_vacuum_t0_sim_L34.txt")
    qw, qwm = R.evolve_circuits(ns["qc_wave_init"], L, t, m, g)
    qv, qvm = R.evolve_circuits(ns["qc_vacuum_init"], L, t, m, g)
    ns.update(qc_wave=qw, qc_wave_mitig=qwm, qc_vacuum=qv, qc_vacuum_mitig=qvm)
    backend = FakeKingston()
    ns["backend"] = backend
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
    isa0 = pm.run(qw)
    layout = list(isa0.layout.initial_index_layout(filter_ancillas=True))
    pm_l = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=layout, seed_transpiler=42)
    ns["circuits_all_isa"] = [pm_l.run(c) for c in (qw, qwm, qv, qvm)]
    ns["observables_isa"] = [o.apply_layout(ns["circuits_all_isa"][0].layout) for o in ns["observables"]]
    ns["qc_isa_init_layout"] = layout
    ns["chain"] = layout
    ns["flight_plan"] = {"canary": {"pubs": 2, "twirls": 16, "shots": 128, "usage_s": 3.8},
                         "main": {"pubs": 4, "twirls": 64, "shots": 256, "usage_s": 31.5}, "cap_s": 180}
    # Part-1 quantities used by the Part-5 error-budget table (values from SPEC section D / test_reference)
    # defined in Part 0 of the real notebook (nb_part_a): the master hardware switch and the
    # contrast of the loaded reference profile
    ns["RUN_ON_HARDWARE"] = False
    _mpsp = "reference_data/mps_reference_L34.npz"
    _Xref = ((np.load(_mpsp)["chi_wave_t8_bd64"] - np.load(_mpsp)["chi_vacuum_t8_bd64"]) if os.path.exists(_mpsp)
             else (np.loadtxt("reference_data/chi_wave_evolved_sim_L34_maxbond40.txt")
                   - np.loadtxt("reference_data/chi_vacuum_evolved_sim_L34_maxbond40.txt")))
    ns["X_ref"] = _Xref
    ns["PEAK_SITES"] = [31, 32, 35, 36]; ns["DIP_SITES"] = [33, 34]
    ns["C_ref"] = float(np.mean(_Xref[[31, 32, 35, 36]]) - np.mean(_Xref[[33, 34]]))
    ns["vqe_fidelity"] = {4: 0.9961, 6: 0.9945, 8: 0.9929}
    ns["truncation_shift"] = 0.012
    ns["trotter_table"] = {(2, 1.0): 0.05, (2, 0.5): 0.013, (2, 0.25): 0.003, (4, 1.0): 0.09, (4, 0.5): 0.024, (4, 0.25): 0.006}
    ns["mps_err"] = {20: 0.3, 40: 0.0228}
    ns["estimator_options"] = {"resilience_level": 0, "default_shots": 64 * 256,
                               "twirling": {"enable_gates": True, "enable_measure": True, "num_randomizations": 64,
                                            "shots_per_randomization": 256, "strategy": "active-accum"},
                               "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"}}
    print(f"namespace ready in {time.time() - t0:.1f} s: CZ = {ns['circuits_all_isa'][0].count_ops().get('cz')}, layout[:4] = {layout[:4]}")
    return ns


def run_cells(ns: dict) -> None:
    timings = []
    total0 = time.time()
    for i, c in enumerate(nb_part_b.CELLS):
        if c["cell_type"] != "code" or "participant-only" in c.get("tags", []):
            continue
        src = nbkit.solution_view(c["source"])
        first = next((ln for ln in src.splitlines() if ln.strip()), "")[:70]
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"CELL {i} FAILED: {first}")
            raise
        finally:
            plt.close("all")
        dt = time.time() - t0
        timings.append((i, dt, first))
        print(f"--- cell {i} OK in {dt:.1f} s :: {first}")
    print("\nper-cell runtime:")
    for i, dt, first in timings:
        print(f"  cell {i:3d}  {dt:7.1f} s  {first}")
    print(f"TOTAL {time.time() - total0:.1f} s over {len(timings)} code cells "
          f"({sum(1 for c in nb_part_b.CELLS if c['cell_type'] == 'markdown')} markdown cells)")
    print("grader calls:", _calls)


if __name__ == "__main__":
    leak_check()
    if "--view" not in sys.argv:
        ns = build_namespace()
        run_cells(ns)
        print("SMOKE TEST PASSED")
