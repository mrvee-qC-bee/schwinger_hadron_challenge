"""
Organizer re-grader for team submissions.

    python organizer/grade_submission.py <path> [<path> ...] [--hidden] [--team NAME] [--accept-quiet-note]
                                         [--leaderboard leaderboard.csv]

<path> is a team's `submission/` folder, the folder that contains it (the unzipped `<team>.zip`), or the
`<team>.zip` itself (extracted to a temporary directory).  One table and one LEADERBOARD line are printed
per team; with --leaderboard the lines are collected into a CSV sorted the way the leaderboard is ranked.

For each team the tool

  1. re-scores every ARRAY-graded exercise from the npz dumps with the same rules as fallfest_grader
     (ex 1.3-1.5, 2.1 layout, 2.2 counts, 2.3 chain, 2.4, 3.3 arrays, 4.1, 4.2),
  2. applies the hardware leaderboard metric (window 25..42, RMSE_W, contrast, quiet, coverage, gates,
     usage <= 180 s, fallback cap 75 %), recomputing X_hat with the team's own `odr_mitigate` when
     `submission/team_functions.py` exists (the notebook's last cells write it) and otherwise with the
     QDC pooled formula (`challenge_utils.postselection_and_mitigation`, which ex 3.1 forces the team's
     function to reproduce to 1e-9),
  3. re-runs the FUNCTION-level exercises (0.x, 1.1, 1.2, 3.1, 3.2) from `team_functions.py`; without that
     file their locally reported points are kept PROVISIONALLY and listed as NOT RE-VERIFIED (check them at
     the viva).  --hidden additionally re-runs `select_chain` on FakeFez and FakeMarrakesh for ex 2.3,
  4. cross-checks score.json against the `[exN] p/m` grader lines printed in the executed notebook
     (a cheap tamper check: a score the notebook never printed, or a different number, is flagged),
  5. prints a table (local vs organizer points), the leaderboard line, and writes organizer_score.json
     next to score.json (for a zip: `<team>.organizer_score.json` next to the zip).

NOT automated (do by hand, ORGANIZER_GUIDE.md section 5): re-fetching job usage and timestamps from IBM
Quantum, and comparing submitted evs with the organizer-released datasets.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]

GRADER_LINE = re.compile(r"^\[(ex\d\.\d|B\d)\]\s+([0-9.]+)/(\d+)")
FUNCTION_LEVEL = ("ex0.1", "ex0.2", "ex0.3", "ex0.4", "ex0.5", "ex1.1", "ex3.1", "ex3.2", "B2", "B3")


def _frozen_backend(name: str):
    """A minimal backend stub carrying only a name, for fallfest_grader.target_summary().
    target_summary raises ValueError when the name has no frozen calibration snapshot."""
    return type("B", (), {"name": name, "target": None})()


def resolve(path: str):
    """-> (team, submission_dir, notebook_path or None, tmpdir or None, zip_path or None)."""
    p = os.path.abspath(path)
    tmp = zip_path = None
    if p.lower().endswith(".zip") and os.path.isfile(p):
        tmp = tempfile.mkdtemp(prefix="ff_regrade_")
        with zipfile.ZipFile(p) as z:
            z.extractall(tmp)
        team, root, zip_path = os.path.splitext(os.path.basename(p))[0], tmp, p
    elif os.path.isdir(p):
        team, root = os.path.basename(p.rstrip(os.sep)), p
    else:
        raise SystemExit(f"{path}: not a directory or a .zip file")
    if os.path.isfile(os.path.join(root, "score.json")):
        sub = root
        if team == "submission":
            team = os.path.basename(os.path.dirname(sub))
    else:
        hits = [h for h in sorted(glob.glob(os.path.join(root, "**", "score.json"), recursive=True)) if "_organizer_rescore" not in h]
        if not hits:
            raise SystemExit(f"{path}: no score.json found (expected <team>/submission/score.json)")
        sub = os.path.dirname(hits[0])
    nb = None
    cands = [os.path.join(os.path.dirname(sub), "schwinger_hadron_participant.ipynb")]
    cands += sorted(glob.glob(os.path.join(root, "**", "schwinger_hadron_participant.ipynb"), recursive=True))
    cands += sorted(glob.glob(os.path.join(root, "**", "*.ipynb"), recursive=True))
    for c in cands:
        if os.path.isfile(c) and "_organizer_rescore" not in c and ".ipynb_checkpoints" not in c:
            nb = c
            break
    return team, sub, nb, tmp, zip_path


def notebook_grader_lines(nb_path: str | None) -> dict | None:
    """{ex: (points, max)} from the LAST '[exN] p/m' line among the executed notebook's outputs."""
    if not nb_path:
        return None
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for cell in nb.get("cells", []):
        for o in cell.get("outputs", []) or []:
            if o.get("output_type") != "stream":
                continue
            text = o.get("text", "")
            text = "".join(text) if isinstance(text, list) else str(text)
            for ln in text.splitlines():
                m = GRADER_LINE.match(ln.strip())
                if m:
                    out[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return out


def grade_one(team: str, team_dir: str, nb_path: str | None, hidden: bool, accept_quiet_note: bool) -> dict:
    import numpy as np
    import fallfest_grader as ff
    import challenge_utils as cu

    ff.SUBMISSION_DIR = os.path.join(team_dir, "_organizer_rescore")
    os.makedirs(ff.SUBMISSION_DIR, exist_ok=True)
    stale = os.path.join(ff.SUBMISSION_DIR, "score.json")
    if os.path.exists(stale):
        os.remove(stale)

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

    # optional team functions (written by the notebook's export cell)
    tf = None
    tf_path = os.path.join(team_dir, "team_functions.py")
    tf_error = None
    if os.path.exists(tf_path):
        try:
            spec = importlib.util.spec_from_file_location(f"team_functions_{abs(hash(team_dir))}", tf_path)
            tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
            print(f"team_functions.py: {len([n for n in dir(tf) if callable(getattr(tf, n)) and not n.startswith('_')])} callables imported")
        except Exception as e:  # noqa: BLE001
            tf, tf_error = None, f"{type(e).__name__}: {e}"
            print(f"team_functions.py could not be imported ({tf_error}): function-level exercises are NOT re-verified")
    else:
        print("no team_functions.py in the submission: function-level exercises are NOT re-verified (kept provisionally)")

    organizer: dict[str, dict] = {}
    notes: dict[str, str] = {}

    def keep_local(ex, why):
        if ex in local:
            organizer[ex] = {"points": local[ex]["points"], "max": local[ex]["max"], "detail": local[ex]["detail"]}
        notes[ex] = why

    def rescore(ex, fn, *a, **kw):
        try:
            fn(*a, **kw)
            organizer[ex] = ff._load_scores()[ex]
        except Exception as e:  # noqa: BLE001
            organizer[ex] = {"points": 0.0, "max": ff.MAX_POINTS[ex], "detail": f"re-score failed: {e}"}
        return organizer[ex]["points"]

    # ---- function-level exercises
    for ex in FUNCTION_LEVEL:
        keep_local(ex, "function-level: local score kept PROVISIONALLY (no team_functions.py) -> viva")
    if tf is not None:
        g = lambda n: getattr(tf, n, None)  # noqa: E731
        if g("chiral_condensate_observables"):
            rescore("ex0.1", ff.grade_ex0_1, g("chiral_condensate_observables")); notes["ex0.1"] = "re-run from team_functions.py"
        if g("RXYplus") and g("RXYminus"):
            rescore("ex0.2", ff.grade_ex0_2, g("RXYplus"), g("RXYminus")); notes["ex0.2"] = "re-run from team_functions.py"
        if g("prep_vacuum"):
            rescore("ex0.3", ff.grade_ex0_3, g("prep_vacuum")); notes["ex0.3"] = "re-run from team_functions.py"
        if g("prep_wave"):
            rescore("ex0.4", ff.grade_ex0_4, g("prep_wave")); notes["ex0.4"] = "re-run from team_functions.py"
        if g("trotter_step") and g("evolve_circuits") and g("prep_wave"):
            rescore("ex0.5", ff.grade_ex0_5, g("trotter_step"), g("evolve_circuits"), g("prep_wave")); notes["ex0.5"] = "re-run from team_functions.py"
        if g("schwinger_hamiltonian") and g("electric_hamiltonian_truncated") and local.get("ex1.1", {}).get("E0_L8") is not None:
            rescore("ex1.1", ff.grade_ex1_1, g("schwinger_hamiltonian"), g("electric_hamiltonian_truncated"), local["ex1.1"]["E0_L8"]); notes["ex1.1"] = "re-run from team_functions.py"
        if g("odr_mitigate") and g("odr_uncertainty") and g("odr_bias"):
            rescore("ex3.1", ff.grade_ex3_1, g("odr_mitigate"), g("odr_uncertainty"), g("odr_bias")); notes["ex3.1"] = "re-run from team_functions.py"
        if g("mitigation_options"):
            rescore("ex3.2", ff.grade_ex3_2, g("mitigation_options")); notes["ex3.2"] = "re-run from team_functions.py"
        if g("barbell_rzz"):
            rescore("B3", ff.grade_bonus_B3, g("barbell_rzz")); notes["B3"] = "re-run from team_functions.py"

    # ---- array-level exercises
    a = arr("ex1.2")
    if a is not None and "truncation_shift" in a and tf is not None and getattr(tf, "electric_layer", None):
        rescore("ex1.2", ff.grade_ex1_2, tf.electric_layer, a["truncation_shift"]); notes["ex1.2"] = "re-run from team_functions.py"
    else:
        keep_local("ex1.2", "electric_layer is a function: local score kept PROVISIONALLY (no team_functions.py) -> viva")
        if a is not None and "truncation_shift" in a and "ex1.2" in organizer:
            ts = np.asarray(a["truncation_shift"], dtype=float); r12 = ff._refs()
            ref = float(r12["truncation_shift_L8_t4"]); ref_arr = np.asarray(r12["truncation_shift_array_L8_t4"], dtype=float)
            ok_ts = (ts.size == 1 and abs(float(ts.ravel()[0]) - ref) < 0.002) or \
                    (ts.shape == (16,) and min(np.max(np.abs(ts - ref_arr)), np.max(np.abs(ts + ref_arr))) < 0.002)
            organizer["ex1.2"]["detail"] = f"{organizer['ex1.2']['detail']} | truncation_shift {'ok' if ok_ts else 'WRONG'} (ref {ref:.4f})"
            if not ok_ts:
                organizer["ex1.2"]["points"] = min(float(organizer["ex1.2"]["points"]), 2.0)
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
        bname21 = local["ex2.1"].get("backend", "fake_kingston")
        d2 = int(a["depth2q"])
        pts = local["ex2.1"]["points"]
        try:
            summ = ff.target_summary(_frozen_backend(bname21))
            ok, why = ff.chain_validity(summ, lay, 68) if len(lay) == 68 else (False, "no 68-qubit layout")
            organizer["ex2.1"] = {"points": pts if ok else 0.0, "max": 3,
                                  "detail": f"local {pts}; layout path {'ok' if ok else 'INVALID: ' + why + ' -> 0'} on {summ.get('name')}, 2q-depth {d2}, cz counts {a['cz_counts'].tolist()}"}
            notes["ex2.1"] = "ISA/observable checks need the circuits (local); layout + depth re-checked"
        except ValueError as e:
            organizer["ex2.1"] = {"points": pts, "max": 3, "detail": f"local {pts}; 2q-depth {d2}; layout NOT re-checked ({e})"}
            notes["ex2.1"] = f"backend {bname21} is not frozen: layout not re-checked"
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
        try:
            summ = ff.target_summary(_frozen_backend(bname))
        except ValueError as e:
            organizer["ex2.3"] = {"points": local["ex2.3"]["points"], "max": 5, "detail": f"local score kept; chain NOT re-checked ({e})"}
            notes["ex2.3"] = f"backend {bname} is not frozen: chain cost not re-checked"
            summ = None
        if summ is not None:
            ok, why = ff.chain_validity(summ, chain, 68)
            c = ff.chain_cost(summ, chain); cb, _ = ff._baseline_for(summ, 68)
            ratio = c / cb
            pts = 0.0 if not ok else 2.0 + (3.0 if ratio <= 1.05 else 2.0 if ratio <= 1.15 else 1.0 if ratio <= 1.30 else 0.0)
            organizer["ex2.3"] = {"points": pts, "max": 5, "detail": f"chain on {bname}: {'valid' if ok else why}; cost {c:.4f} vs baseline {cb:.4f} (ratio {ratio:.3f})"}
            notes["ex2.3"] = "chain validity and cost re-checked from the stored chain"
        if hidden:
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
    elif "ex4.1" in local:
        organizer["ex4.1"] = {"points": 0.0, "max": 3, "detail": "no canary arrays in ex4_1.npz (no data) -> 0"}; notes["ex4.1"] = "no canary data"
    hw = None
    a = arr("ex4.2")
    if a is not None and "ex4.2" in local:
        ji = local["ex4.2"].get("job_info", {})
        hr = {k: v for k, v in a.items() if v.size == 68}
        hr["fallback"] = bool(local["ex4.2"].get("fallback", False))
        odr_fn = getattr(tf, "odr_mitigate", None) if tf is not None else None
        if callable(odr_fn):
            rescore("ex4.2", ff.grade_ex4_2, hr, ji, odr_fn, accept_quiet_note)
            notes["ex4.2"] = "X_hat recomputed with the team's odr_mitigate (team_functions.py)"
        else:
            rescore("ex4.2", ff.grade_ex4_2, hr, ji, cu.postselection_and_mitigation, accept_quiet_note)
            notes["ex4.2"] = "X_hat recomputed with the QDC pooled formula (no team_functions.py; ex 3.1 forces equality)"
        hw = ff._load_scores()["ex4.2"]
    elif "ex4.2" in local:
        organizer["ex4.2"] = {"points": 0.0, "max": 18, "detail": "no hardware arrays in ex4_2.npz (no data) -> 0"}; notes["ex4.2"] = "no hardware data"
    for ex in ("ex4.3", "B1"):
        keep_local(ex, "improvement dict / bonus arrays: local score kept, judge reviews")

    # ---- tamper check: score.json vs the grader lines the executed notebook printed
    tamper = []
    nb_lines = notebook_grader_lines(nb_path)
    if nb_lines is None:
        tamper.append("no executed notebook found next to submission/ -- score.json could not be cross-checked")
    else:
        for ex, rec in local.items():
            if ex not in ff.MAX_POINTS or not isinstance(rec, dict):
                continue
            if ex not in nb_lines:
                tamper.append(f"TAMPER? score.json has {ex} = {float(rec['points']):.1f} but the executed notebook shows no grader line for it")
            elif abs(nb_lines[ex][0] - float(rec["points"])) > 0.051:
                tamper.append(f"TAMPER? score.json {ex} = {float(rec['points']):.1f} but the notebook's last grader line shows {nb_lines[ex][0]:.1f}/{nb_lines[ex][1]}")

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
    unverified = [ex for ex, why in notes.items() if "PROVISIONALLY" in why and ex in organizer]
    if unverified:
        print(f"\nNOT RE-VERIFIED (points taken from the team's score.json; check at the viva): {', '.join(unverified)}"
              f"  [{sum(organizer[e]['points'] for e in unverified):.1f} points]"
              + (f" -- team_functions.py import error: {tf_error}" if tf_error else ""))
    flags = list(hw.get("flags") or []) if hw else []
    for ex, rec in organizer.items():
        for m in re.finditer(r"(JUDGE FLAG|FLAG)[^;|]*", str(rec.get("detail", ""))):
            s = f"{ex}: {m.group(0).strip()}"
            if s not in flags:
                flags.append(s)
    flags += tamper
    if flags:
        print("FLAGS:")
        for f in flags:
            print("   -", f)
    rmse_w = hw.get("rmse_w") if hw else None
    # Rank on the NaN-penalised RMSE: a site the team did not report counts as "no signal", so
    # raising the post-selection threshold can never buy a better rank (see fallfest_grader.hardware_metric).
    rmse_rank = hw.get("rmse_w_rank", rmse_w) if hw else None
    n_fin = hw.get("n_finite_in_window") if hw else None
    usage = hw.get("usage_s") if hw else None
    r68 = hw.get("rmse_68") if hw else None
    eligible = n_fin is None or n_fin >= 16
    line = (f"LEADERBOARD | {team} | total {total:.1f}/90 auto (+{bonus:.1f} bonus, ex5 by judges) "
            f"| RANK-RMSE {'-' if rmse_rank is None else f'{rmse_rank:.3f}'} | RMSE_W {'-' if rmse_w is None else f'{rmse_w:.3f}'} "
            f"| RMSE_68 {'-' if r68 is None else f'{r68:.3f}'} | usage {usage} s | fallback {hw.get('fallback') if hw else None}"
            f" | flags {len(flags)} | not re-verified {len(unverified)}"
            + ("" if eligible else f" | NOT RANKED: only {n_fin}/18 window sites reported"))
    print("\n" + line)
    out = {"team": team, "total_auto": total, "bonus": bonus, "rmse_w": rmse_w, "rmse_w_rank": rmse_rank,
           "n_finite_in_window": n_fin, "leaderboard_eligible": bool(eligible),
           "rmse_68": r68, "usage_s": usage, "fallback": hw.get("fallback") if hw else None,
           "hidden": hidden, "organizer": organizer, "notes": notes, "not_reverified": unverified,
           "flags": flags, "tamper": tamper, "notebook_checked": nb_path is not None and nb_lines is not None}
    with open(os.path.join(team_dir, "organizer_score.json"), "w") as f:
        json.dump(ff._jsonable(out), f, indent=1)
    print(f"wrote {os.path.join(team_dir, 'organizer_score.json')}")
    return out


LEADERBOARD_COLS = ["rank", "team", "rmse_w_rank", "rmse_w", "rmse_68", "usage_s", "fallback", "leaderboard_eligible",
                    "total_auto", "bonus", "not_reverified", "n_flags", "hidden"]


def write_leaderboard(results: list[dict], path: str) -> None:
    """Merge this run's teams into the CSV (existing rows of other teams are kept) and sort as ranked:
    genuine before fallback rows, ranked before NOT RANKED, then RANK-RMSE, RMSE_68, lower usage."""
    rows: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["team"]] = r
    for res in results:
        rows[res["team"]] = {"team": res["team"], "rmse_w_rank": res["rmse_w_rank"], "rmse_w": res["rmse_w"], "rmse_68": res["rmse_68"],
                             "usage_s": res["usage_s"], "fallback": res["fallback"], "leaderboard_eligible": res["leaderboard_eligible"],
                             "total_auto": round(res["total_auto"], 2), "bonus": res["bonus"], "not_reverified": " ".join(res["not_reverified"]),
                             "n_flags": len(res["flags"]), "hidden": res["hidden"]}

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    def key(r):
        fb = str(r.get("fallback")).lower() in ("true", "1")
        el = str(r.get("leaderboard_eligible")).lower() not in ("false", "0")
        return (fb, not el, num(r.get("rmse_w_rank")), num(r.get("rmse_68")), num(r.get("usage_s")))

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLS, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(sorted(rows.values(), key=key), 1):
            r = {k: ("" if v is None else v) for k, v in r.items()}; r["rank"] = i
            w.writerow(r)
    print(f"leaderboard: {path} ({len(rows)} teams)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Organizer re-grader (see the module docstring).")
    ap.add_argument("paths", nargs="+", help="<team>.zip, the unzipped team folder, or its submission/ folder (several allowed)")
    ap.add_argument("--hidden", action="store_true", help="also re-run select_chain on FakeFez/FakeMarrakesh (needs team_functions.py)")
    ap.add_argument("--team", default=None, help="team name for the report (single path only; default: folder/zip name)")
    ap.add_argument("--accept-quiet-note", action="store_true", help="judge accepted the systematics note for the quiet criterion")
    ap.add_argument("--leaderboard", default=None, metavar="CSV", help="merge the LEADERBOARD lines into this CSV, sorted as ranked")
    args = ap.parse_args()
    if args.team and len(args.paths) > 1:
        ap.error("--team applies to a single path")
    results = []
    for path in args.paths:
        team, sub, nb, tmp, zip_path = resolve(path)
        if args.team:
            team = args.team
        print(f"\n===== {team}: {sub}" + (f"  (notebook: {os.path.basename(nb)})" if nb else "  (no executed notebook found)"))
        try:
            res = grade_one(team, sub, nb, args.hidden, args.accept_quiet_note)
            results.append(res)
            if zip_path:
                keep = os.path.join(os.path.dirname(zip_path), f"{team}.organizer_score.json")
                shutil.copy(os.path.join(sub, "organizer_score.json"), keep)
                print(f"copied to {keep}")
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    if args.leaderboard:
        write_leaderboard(results, args.leaderboard)


if __name__ == "__main__":
    main()
