"""
Organizer re-grader for one team submission folder.

    python organizer/grade_submission.py <team_submission_dir> [--hidden] [--team NAME] [--accept-quiet-note]

<team_submission_dir> is the team's `submission/` folder: score.json (written by fallfest_grader) plus the
exN.npz array dumps.  This tool

  1. re-scores every ARRAY-graded exercise from the npz dumps with the same rules as fallfest_grader
     (ex 1.2-1.5, 2.1, 2.2 counts, 2.3 chain, 2.4, 3.3, 4.1, 4.2), using the organizer's own ODR
     (challenge_utils.postselection_and_mitigation, i.e. the QDC pooled formula that ex 3.1 forces the team's
     odr_mitigate to reproduce to 1e-9) for the hardware metric,
  2. applies the hardware leaderboard metric (window 25..42, RMSE_W, contrast, quiet, coverage, gates,
     usage <= 180 s, fallback cap 75 %),
  3. keeps the locally reported points for FUNCTION-level exercises (0.x, 1.1, 3.1, 3.2, 2.2 matched circuits,
     2.3 search) unless `<dir>/team_functions.py` exists, in which case those functions are re-imported and
     re-run (with --hidden: select_chain on FakeFez and FakeMarrakesh, ODR on fresh synthetic seeds),
  4. prints a table (local vs organizer points), a leaderboard line `team | total | RMSE_W | usage`, and
     writes <dir>/organizer_score.json.

Limitation (documented in SPEC C / BUILD_NOTES): participant functions live in the notebook, so hidden-mode
function re-runs need the team to export them to team_functions.py; otherwise the viva covers them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("team_dir")
    ap.add_argument("--hidden", action="store_true", help="re-run function checks on FakeFez/FakeMarrakesh + fresh seeds (needs team_functions.py)")
    ap.add_argument("--team", default=None)
    ap.add_argument("--accept-quiet-note", action="store_true", help="judge accepted the systematics note for the quiet criterion")
    args = ap.parse_args()
    team_dir = os.path.abspath(args.team_dir)
    team = args.team or os.path.basename(team_dir.rstrip("/"))
    os.environ["FF_SUBMISSION_DIR"] = os.path.join(team_dir, "_organizer_rescore")
    sys.path[:0] = [ROOT]
    import numpy as np
    import fallfest_grader as ff
    import challenge_utils as cu

    local_path = os.path.join(team_dir, "score.json")
    local = json.load(open(local_path)) if os.path.exists(local_path) else {}

    def arr(ex):
        p = os.path.join(team_dir, f"{ex.replace('.', '_')}.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=False)
        return {k: d[k] for k in d.files}

    def js(a, key):
        return json.loads(str(a[key])) if a is not None and key in a else None

    # optional team functions (exported by the team)
    tf = None
    tf_path = os.path.join(team_dir, "team_functions.py")
    if os.path.exists(tf_path):
        spec = importlib.util.spec_from_file_location("team_functions", tf_path)
        tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
        print(f"team_functions.py found: {[n for n in dir(tf) if not n.startswith('_')]}")

    organizer: dict[str, dict] = {}
    notes: dict[str, str] = {}

    def keep_local(ex, why):
        if ex in local:
            organizer[ex] = {"points": local[ex]["points"], "max": local[ex]["max"], "detail": local[ex]["detail"]}
        notes[ex] = why

    def rescore(ex, fn, *a, **kw):
        try:
            pts = fn(*a, **kw)
            organizer[ex] = ff._load_scores()[ex]
        except Exception as e:  # noqa: BLE001
            organizer[ex] = {"points": 0.0, "max": ff.MAX_POINTS[ex], "detail": f"re-score failed: {e}"}
        return organizer[ex]["points"]

    # ---- function-level exercises
    for ex in ("ex0.1", "ex0.2", "ex0.3", "ex0.4", "ex0.5", "ex1.1", "ex3.1", "ex3.2", "B2", "B3"):
        keep_local(ex, "function-level: local score kept (viva)")
    if tf is not None:
        g = lambda n: getattr(tf, n, None)  # noqa: E731
        if all(g(n) for n in ("chiral_condensate_observables",)):
            rescore("ex0.1", ff.grade_ex0_1, g("chiral_condensate_observables")); notes["ex0.1"] = "re-run from team_functions.py"
        if g("RXYplus") and g("RXYminus"):
            rescore("ex0.2", ff.grade_ex0_2, g("RXYplus"), g("RXYminus")); notes["ex0.2"] = "re-run"
        if g("prep_vacuum"):
            rescore("ex0.3", ff.grade_ex0_3, g("prep_vacuum")); notes["ex0.3"] = "re-run"
        if g("prep_wave"):
            rescore("ex0.4", ff.grade_ex0_4, g("prep_wave")); notes["ex0.4"] = "re-run"
        if g("trotter_step") and g("evolve_circuits") and g("prep_wave"):
            rescore("ex0.5", ff.grade_ex0_5, g("trotter_step"), g("evolve_circuits"), g("prep_wave")); notes["ex0.5"] = "re-run"
        if g("schwinger_hamiltonian") and g("electric_hamiltonian_truncated") and local.get("ex1.1", {}).get("E0_L8") is not None:
            rescore("ex1.1", ff.grade_ex1_1, g("schwinger_hamiltonian"), g("electric_hamiltonian_truncated"), local["ex1.1"]["E0_L8"]); notes["ex1.1"] = "re-run"
        if g("odr_mitigate") and g("odr_uncertainty") and g("odr_bias"):
            rescore("ex3.1", ff.grade_ex3_1, g("odr_mitigate"), g("odr_uncertainty"), g("odr_bias")); notes["ex3.1"] = "re-run"
        if g("twirl_circuit") and g("postselect_charge"):
            rescore("ex3.2", ff.grade_ex3_2, g("twirl_circuit"), g("postselect_charge")); notes["ex3.2"] = "re-run"

    # ---- array-level exercises
    a = arr("ex1.2")
    if a is not None and "truncation_shift" in a and tf is not None and getattr(tf, "electric_layer", None):
        rescore("ex1.2", ff.grade_ex1_2, tf.electric_layer, a["truncation_shift"]); notes["ex1.2"] = "re-run"
    else:
        keep_local("ex1.2", "electric_layer is a function: local score kept; truncation_shift re-checked" if a is None else
                   f"truncation_shift re-checked: {float(np.asarray(a['truncation_shift']).ravel()[0]) if np.asarray(a['truncation_shift']).ndim == 0 or np.asarray(a['truncation_shift']).size == 1 else 'array'}")
    a = arr("ex1.3")
    if a is not None:
        rescore("ex1.3", ff.grade_ex1_3, {int(k): v for k, v in js(a, "vqe_fidelity").items()}, {int(k): v for k, v in js(a, "vqe_energy_gap").items()})
    a = arr("ex1.4")
    if a is not None and "ex1.4" in local:
        tab = {tuple(float(x) for x in k.split(",")): v for k, v in js(a, "trotter_table").items()}
        rescore("ex1.4", ff.grade_ex1_4, tab, float(a["richardson_error"]), str(local["ex1.4"].get("dominant_error", "")))
    a = arr("ex1.5")
    if a is not None:
        rescore("ex1.5", ff.grade_ex1_5, {int(k): v for k, v in js(a, "mps_err").items()}, js(a, "mps_seconds"), a["X_bond40"])
    a = arr("ex2.1")
    if a is not None and "ex2.1" in local:
        lay = [int(q) for q in a["layout"]]
        summ = ff.target_summary(type("B", (), {"name": local["ex2.1"].get("backend", "fake_kingston"), "target": None})())
        ok, why = ff.chain_validity(summ, lay, 68) if len(lay) == 68 else (False, "no 68-qubit layout")
        d2 = int(a["depth2q"])
        pts = local["ex2.1"]["points"]
        organizer["ex2.1"] = {"points": pts, "max": 3, "detail": f"local {pts}; layout path {'ok' if ok else why} on {summ.get('name')}, 2q-depth {d2}, cz counts {a['cz_counts'].tolist()}"}
        notes["ex2.1"] = "ISA/observable checks need the circuits (local); layout + depth re-checked"
    a = arr("ex2.2")
    if a is not None:
        r = ff._refs(); n2 = int(a["n2q_logical"]); ncz = int(a["n_cz_physics"])
        p = (1.0 if abs(n2 - int(r["n2q_logical_t8"])) <= 0.01 * int(r["n2q_logical_t8"]) else 0.0) + \
            (1.0 if abs(ncz - int(r["n_cz_physics_t8_kingston_O3_seed42"])) <= 0.01 * int(r["n_cz_physics_t8_kingston_O3_seed42"]) else
             (0.5 if abs(ncz - int(r["n_cz_physics_t8_kingston_O1_seed42"])) <= 0.01 * int(r["n_cz_physics_t8_kingston_O1_seed42"]) else 0.0))
        lp = local.get("ex2.2", {}).get("points", 0.0)
        organizer["ex2.2"] = {"points": min(lp, p + 3.0), "max": 5, "detail": f"counts re-checked ({p}/2); matched-circuit part from local score"}
        notes["ex2.2"] = "matched circuits need evolve_circuits_matched (local/viva)"
    a = arr("ex2.3")
    if a is not None and "ex2.3" in local:
        chain = [int(q) for q in a["chain"]]
        bname = local["ex2.3"].get("backend", "fake_kingston")
        summ = ff.target_summary(type("B", (), {"name": bname, "target": None})())
        ok, why = ff.chain_validity(summ, chain, 68)
        c = ff.chain_cost(summ, chain); cb, _ = ff._baseline_for(summ, 68)
        ratio = c / cb
        pts = 0.0 if not ok else 2.0 + (3.0 if ratio <= 1.05 else 2.0 if ratio <= 1.15 else 1.0 if ratio <= 1.30 else 0.0)
        organizer["ex2.3"] = {"points": pts, "max": 5, "detail": f"chain on {bname}: {'valid' if ok else why}; cost {c:.4f} vs baseline {cb:.4f} (ratio {ratio:.3f})"}
        if args.hidden:
            if tf is not None and getattr(tf, "select_chain", None):
                from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh
                hidden_pts = []
                for B in (FakeFez, FakeMarrakesh):
                    hidden_pts.append(rescore("ex2.3", ff.grade_ex2_3, tf.select_chain, B()))
                organizer["ex2.3"] = {"points": float(np.mean(hidden_pts)), "max": 5, "detail": f"hidden mode: FakeFez/FakeMarrakesh points {hidden_pts} (mean)"}
                notes["ex2.3"] = "hidden: select_chain re-run on FakeFez + FakeMarrakesh"
            else:
                notes["ex2.3"] = "hidden mode requested but no team_functions.select_chain: ask the team to run select_chain(FakeFez()) at the viva"
    a = arr("ex2.4")
    if a is not None and "ex2.4" in local:
        fp = js(a, "flight_plan")
        try:
            execs = 4 * int(fp["num_randomizations"]) * int(fp["shots_per_randomization"]); usage = 2 + 0.45e-3 * execs
            organizer["ex2.4"] = {"points": local["ex2.4"]["points"], "max": 2, "detail": f"usage model {usage:.1f} s ({execs} executions); local {local['ex2.4']['points']}"}
        except Exception as e:  # noqa: BLE001
            organizer["ex2.4"] = {"points": 0.0, "max": 2, "detail": f"flight plan unreadable: {e}"}
        notes["ex2.4"] = "EstimatorOptions object not stored: local score kept"
    a = arr("ex3.3")
    if a is not None and "ex3.3" in local:
        r = ff._refs(); Xex = r["X_trotter_L6_t4"]
        rm = lambda x: float(np.sqrt(np.nanmean((x - Xex) ** 2)))  # noqa: E731
        er, em = rm(a["X_raw"]), rm(a["X_mit"])
        cw_ex = r["chi_wave_t0_L6"]
        xm = cu.postselection_and_mitigation(a["chi_wave"], a["chi_wave_mitig"], cw_ex, 6) - cu.postselection_and_mitigation(a["chi_vacuum"], a["chi_vacuum_mitig"], r["chi_vacuum_t0_L6"], 6)
        xm2 = cu.postselection_and_mitigation(a["chi_wave"], a["chi_wave_mitig"], cw_ex, 6) - cu.postselection_and_mitigation(a["chi_vacuum"], a["chi_vacuum_mitig"], r["chi_vacuum_sub_t0_L6"], 6)
        consistent = np.allclose(xm, a["X_mit"], atol=1e-6, equal_nan=True) or np.allclose(xm2, a["X_mit"], atol=1e-6, equal_nan=True)
        f = a["factors"]; n_in = int(np.sum((f > 0.02) & (f < 0.97)))
        p_arr = (2.0 if consistent else 0.0) + (2.0 if (consistent and em <= 0.8 * er and em <= 0.10) else 0.0) + (1.0 if n_in >= 8 else 0.0)
        lp = local["ex3.3"]["points"]
        organizer["ex3.3"] = {"points": min(lp, p_arr + 3.0), "max": 8, "detail": f"arrays: X_mit {'consistent' if consistent else 'INCONSISTENT'} with the QDC ODR, RMSE raw {er:.4f} -> mit {em:.4f}, factors in range {n_in}/12; noise-model/chain part from local"}
        notes["ex3.3"] = "noise model object not stored: its 3 points from local score"
    a = arr("ex4.1")
    if a is not None and "ex4.1" in local and a["chi_wave"].size == 68:
        ji = local["ex4.1"].get("job_info", {})
        cr = {"chi_wave": a["chi_wave"], "chi_vacuum": a["chi_vacuum"], "fallback": local["ex4.1"].get("fallback", False)}
        ew, evac = ff._qdc_t0_exact(); ret = (1 - a["chi_vacuum"]) / (1 - evac)
        cr["flagged_sites"] = [int(j) for j in np.where(ret < 0.4)[0]]; cr["verdict"] = "re-scored"
        rescore("ex4.1", ff.grade_ex4_1, cr, ji); notes["ex4.1"] = "flagged sites/verdict taken from the retention rule"
    hw = None
    a = arr("ex4.2")
    if a is not None and "ex4.2" in local:
        ji = local["ex4.2"].get("job_info", {})
        hr = {k: v for k, v in a.items() if v.size == 68}
        hr["fallback"] = bool(local["ex4.2"].get("fallback", False))
        rescore("ex4.2", ff.grade_ex4_2, hr, ji, cu.postselection_and_mitigation, args.accept_quiet_note)
        hw = ff._load_scores()["ex4.2"]
        notes["ex4.2"] = "organizer ODR = QDC pooled formula (team's odr_mitigate must equal it per ex 3.1)"
    for ex in ("ex4.3", "B1"):
        keep_local(ex, "improvement dict / bonus arrays: local score kept, judge reviews")

    # ---- report
    total = 0.0; bonus = 0.0
    print(f"\n{'ex':7s} {'local':>6s} {'organ.':>7s}  note / detail")
    for ex in ff.MAX_POINTS:
        lp = local.get(ex, {}).get("points")
        op = organizer.get(ex, {}).get("points")
        det = organizer.get(ex, {}).get("detail", "")
        print(f"{ex:7s} {('-' if lp is None else f'{lp:.1f}'):>6s} {('-' if op is None else f'{op:.1f}'):>7s}  {notes.get(ex, '')[:50]:50s} {det[:80]}")
        if op is not None:
            if ex.startswith("B"):
                bonus += op
            else:
                total += op
    rmse_w = hw.get("rmse_w") if hw else None
    usage = hw.get("usage_s") if hw else None
    r68 = hw.get("rmse_68") if hw else None
    line = (f"LEADERBOARD | {team} | total {total:.1f}/90 auto (+{bonus:.1f} bonus, ex5 by judges) | RMSE_W {'-' if rmse_w is None else f'{rmse_w:.3f}'} "
            f"| RMSE_68 {'-' if r68 is None else f'{r68:.3f}'} | usage {usage} s | fallback {hw.get('fallback') if hw else None}")
    print("\n" + line)
    out = {"team": team, "total_auto": total, "bonus": bonus, "rmse_w": rmse_w, "rmse_68": hw.get("rmse_68") if hw else None, "usage_s": usage,
           "hidden": args.hidden, "organizer": organizer, "notes": notes, "flags": hw.get("flags") if hw else None}
    with open(os.path.join(team_dir, "organizer_score.json"), "w") as f:
        json.dump(ff._jsonable(out), f, indent=1)
    print(f"wrote {os.path.join(team_dir, 'organizer_score.json')}")


if __name__ == "__main__":
    main()
