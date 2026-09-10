"""
Organizer tool: bond-32 MPS profiles at L=34, t=8 for dt = 1 (8 steps), 0.5 (16 steps) and 0.25 (32 steps),
used by Bonus B1 (Richardson dt -> 0) so that the solution notebook does not have to run the
multi-minute MPS simulations itself.

    python organizer/make_bonus_b1_reference.py        # writes reference_data/mps_reference_L34_dt.npz

Keys: chi_{wave,vacuum}_t8_dt{1.0,0.5,0.25}_bd32 (same bond dimension for all dt, so that differences are pure Trotter error)
(+ the same at bond 64 when --bond64 is given), seconds_* wall times.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import schwinger_reference as R  # noqa: E402
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2  # noqa: E402


def main() -> None:
    bonds = [32] + ([64] if "--bond64" in sys.argv else [])
    L, t = 34, 8.0
    obs = R.chiral_condensate_observables(L)
    qw, qv = R.prep_wave(L), R.prep_vacuum_for_subtraction(L)
    out_path = os.path.join(ROOT, "reference_data", "mps_reference_L34_dt.npz")
    out = dict(np.load(out_path)) if os.path.exists(out_path) else {}
    for bd in bonds:
        est = AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state",
                                                          "matrix_product_state_max_bond_dimension": bd,
                                                          "matrix_product_state_truncation_threshold": 1e-10},
                                      "run_options": {"seed_simulator": 7}})
        for dt in (1.0, 0.5, 0.25):
            n_steps = int(round(t / dt))
            qpw, _ = R.evolve_circuits(qw, L, t, n_steps=n_steps)
            qpv, _ = R.evolve_circuits(qv, L, t, n_steps=n_steps)
            t0 = time.time()
            r = est.run([(qpw.decompose(reps=3), obs), (qpv.decompose(reps=3), obs)]).result()
            dts = time.time() - t0
            out[f"chi_wave_t8_dt{dt}_bd{bd}"] = np.asarray(r[0].data.evs, float)
            out[f"chi_vacuum_t8_dt{dt}_bd{bd}"] = np.asarray(r[1].data.evs, float)
            out[f"seconds_dt{dt}_bd{bd}"] = np.float64(dts)
            X = out[f"chi_wave_t8_dt{dt}_bd{bd}"] - out[f"chi_vacuum_t8_dt{dt}_bd{bd}"]
            print(f"dt={dt} ({n_steps} steps) bond {bd}: {dts:.0f} s, centre X = {np.round(X[32:36], 3)}", flush=True)
            np.savez(out_path, **out)
    print("wrote", out_path, sorted(out))


if __name__ == "__main__":
    main()
