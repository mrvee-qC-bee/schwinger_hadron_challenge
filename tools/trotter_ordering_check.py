"""
Trotter-ordering check (ORGANIZER_GUIDE.md section 12): compare the pinned odd-first kinetic sublattice
ordering (Fig. 8 of arXiv:2401.08044, `schwinger_reference.trotter_step`) with the even-first alternative
at L = 34, t = 8, bond dimension 40, against the QDC bond-40 file.

    MPLBACKEND=Agg python tools/trotter_ordering_check.py        # about 1 min

Expected: pinned odd-first max|X - X_QDC40| < 1e-4 (bit-identical up to MPS noise); even-first about 0.023.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "organizer")]
import schwinger_reference as R  # noqa: E402
import challenge_utils as cu  # noqa: E402
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2  # noqa: E402


def trotter_step_even_first(qc, L, dt, m=R.M_DEFAULT, g=R.G_DEFAULT):
    """The even-first alternative: even bonds (0,1),(2,3),... open the first kinetic half-step."""
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


def main():
    L, obs = 34, R.chiral_condensate_observables(34)
    rd = os.path.join(ROOT, "reference_data")
    Xq = np.loadtxt(f"{rd}/chi_wave_evolved_sim_L34_maxbond40.txt") - np.loadtxt(f"{rd}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
    est = AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": 40},
                                  "run_options": {"seed_simulator": 7}})
    for name, step in [("pinned odd-first (Fig. 8)", R.trotter_step), ("alternative even-first", trotter_step_even_first)]:
        outs = []
        for init in [R.prep_wave(L), R.prep_vacuum_for_subtraction(L)]:
            qc = init.copy()
            for _ in range(8):
                qc = step(qc, L, 1.0)
            outs.append(qc.decompose(reps=3))
        t0 = time.time()
        r = est.run([(outs[0], obs), (outs[1], obs)]).result()
        X = r[0].data.evs - r[1].data.evs
        print(f"{name}: {time.time() - t0:.0f} s  max|X - X_QDC40| = {np.max(np.abs(X - Xq)):.5f}  "
              f"nrmse = {np.linalg.norm(X - Xq) / np.linalg.norm(Xq):.4f}  centre = {np.round(X[31:37], 3)}", flush=True)


if __name__ == "__main__":
    main()
