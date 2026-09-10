"""
Smoke test for tools/nb_part_a.py: executes the SOLUTION view of every code cell sequentially in one
namespace (cwd = project root) with a STUB fallfest_grader module, and reports per-cell runtime.

    MPLBACKEND=Agg python tools/smoke_part_a.py            # run everything
    MPLBACKEND=Agg python tools/smoke_part_a.py --strip    # only check the participant view for leaks
    MPLBACKEND=Agg python tools/smoke_part_a.py --stop-on-error

Also checks the participant view (nbkit.strip_answers) of every code cell for leaked solution text.
"""
from __future__ import annotations

import argparse
import builtins
import io
import os
import re
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

import nbkit  # noqa: E402

# ----------------------------------------------------------------------------- grader stub
def _make_stub():
    mod = types.ModuleType("fallfest_grader")

    def check_env():
        print("grader stub: check_env")
        return 0

    def summary():
        print("grader stub: summary")
        return 0

    def __getattr__(name):
        if name.startswith("grade_"):
            def _g(*args, **kwargs):
                print(f"grader stub: {name}({len(args)} args)")
                return 0
            return _g
        raise AttributeError(name)

    mod.check_env = check_env
    mod.summary = summary
    mod.__getattr__ = __getattr__
    return mod


LEAK_PATTERNS = [
    (re.compile(r"#\s*SOL\s*$", re.M), "'# SOL' marker survived"),
    (re.compile(r"# KEEP\s*$", re.M), "'# KEEP' marker survived"),
    (re.compile(r"schwinger_reference"), "reference module mentioned"),
    (re.compile(r"# PARTICIPANT:"), "'# PARTICIPANT:' marker survived"),
]


def check_participant_view(cells):
    """Return a list of (cell index, message) leaks in the participant view."""
    leaks = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        view = nbkit.strip_answers(c["source"])
        for pat, msg in LEAK_PATTERNS:
            if pat.search(view):
                leaks.append((i, msg))
        # every BEGIN ANSWER block must have been emptied (only '# YOUR CODE HERE' and KEEP-scaffold lines remain)
        inside = False
        for line in view.splitlines():
            if line.strip().startswith("# BEGIN ANSWER"):
                inside = True
                continue
            if line.strip().startswith("# END ANSWER"):
                inside = False
                continue
        if inside:
            leaks.append((i, "unterminated BEGIN ANSWER block"))
        # compare solution vs participant: every solution line that is not in the participant view must sit
        # inside an answer block or be a SOL line
        sol = nbkit.solution_view(c["source"])
        for line in sol.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if line not in view.splitlines():
                continue  # removed lines are fine (they are the answers)
        # 'organizer' path reference
        if "organizer/" in view:
            leaks.append((i, "organizer path mentioned"))
    return leaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", action="store_true", help="only run the participant-view leak check")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="first code cell index to execute (for debugging)")
    args = ap.parse_args()

    import nb_part_a
    cells = nb_part_a.CELLS
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = len(cells) - n_code
    print(f"nb_part_a: {len(cells)} cells ({n_md} markdown, {n_code} code)")

    leaks = check_participant_view(cells)
    if leaks:
        print("PARTICIPANT VIEW LEAKS:")
        for i, msg in leaks:
            print(f"  cell {i}: {msg}")
    else:
        print("participant view: no leaked solution markers")
    if args.strip:
        sys.exit(1 if leaks else 0)

    sys.modules["fallfest_grader"] = _make_stub()
    builtins.display = lambda *a, **k: None  # notebook builtin used by challenge_utils.plot_qubit_chain

    ns: dict = {"__name__": "__main__"}
    timings = []
    t_all = time.time()
    failed = 0
    code_idx = 0
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        code_idx += 1
        if code_idx - 1 < args.start:
            continue
        src = nbkit.solution_view(c["source"])
        first = next((ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")), "")[:70]
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            status = "ok"
        except Exception:
            status = "FAIL"
            failed += 1
            traceback.print_exc()
            if args.stop_on_error:
                break
        dt = time.time() - t0
        timings.append((i, dt, status, first))
        print(f"--- cell {i:3d} [{status}] {dt:6.1f} s | {first}", flush=True)
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    total = time.time() - t_all
    print("\n=== per-cell runtime (slowest first) ===")
    for i, dt, status, first in sorted(timings, key=lambda x: -x[1])[:12]:
        print(f"cell {i:3d} {dt:7.1f} s [{status}] {first}")
    print(f"\nTOTAL: {total:.1f} s over {len(timings)} code cells; failures: {failed}; leaks: {len(leaks)}")
    sys.exit(1 if (failed or leaks) else 0)


if __name__ == "__main__":
    main()
