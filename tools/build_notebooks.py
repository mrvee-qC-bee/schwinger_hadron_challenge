"""
Build the participant and solution notebooks from the single cell source
(tools/nb_part_a.py + tools/nb_part_b.py), optionally executing the solution.

    python tools/build_notebooks.py            # write both notebooks
    python tools/build_notebooks.py --execute  # also execute the solution notebook in place
    python tools/build_notebooks.py --execute --timeout 5400
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

import nbkit  # noqa: E402

PARTICIPANT = os.path.join(ROOT, "schwinger_hadron_participant.ipynb")
SOLUTION = os.path.join(ROOT, "schwinger_hadron_solution.ipynb")


def all_cells() -> list[dict]:
    import nb_part_a
    import nb_part_b
    return list(nb_part_a.CELLS) + list(nb_part_b.CELLS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute the solution notebook in place")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout (s) for execution")
    ap.add_argument("--kernel", default="python3", help="kernelspec name (the builder's interpreter is put first on PATH)")
    args = ap.parse_args()

    cells = all_cells()
    nb_p = nbkit.build_notebook(cells, participant=True)
    nb_s = nbkit.build_notebook(cells, participant=False)
    nbkit.write_notebook(nb_p, PARTICIPANT)
    nbkit.write_notebook(nb_s, SOLUTION)
    print(f"wrote {PARTICIPANT} ({len(nb_p.cells)} cells)")
    print(f"wrote {SOLUTION} ({len(nb_s.cells)} cells)")

    if args.execute:
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
            "--ExecutePreprocessor.record_timing=True",
            SOLUTION,
        ]
        print("executing:", " ".join(cmd))
        # The default "python3" kernelspec launches a bare `python`; put the interpreter running this
        # builder first on PATH so the kernel is the same environment (qiskit, matplotlib, ...).
        env = dict(os.environ,
                   MPLBACKEND="Agg",
                   PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""),
                   PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        print(f"solution notebook executed OK in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
