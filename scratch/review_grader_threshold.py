"""Leaderboard gaming: a legitimate odr_mitigate whose DEFAULT suppression_threshold is high NaNs the low-retention window
sites; RMSE_W (the leaderboard ranking metric) has no NaN penalty."""
from review_grader_common import setup, run, ROOT
import os, numpy as np
ff = setup("thr")
import schwinger_reference as R
ew, evac = ff._qdc_t0_exact(); hw = np.load(os.path.join(ROOT, "reference_data", "hardware_ibm_kingston_2026-07-25.npz"))
sgn = np.array([(-1) ** j for j in range(68)])
d = {}
for k, key in (("chi_wave", "z_wave"), ("chi_wave_mitig", "z_mitig_wave"), ("chi_vacuum", "z_vacuum"), ("chi_vacuum_mitig", "z_mitig_vacuum")):
    d[k] = sgn * hw[f"odr__T8__{key}"] + 1; d[k + "_std"] = hw[f"odr__T8__{key}_std"]
job = {"job_id": "d9hr3p50k0jc738il8dg", "backend": "ibm_kingston", "usage_s": 96.0, "layout": list(hw["initial_layout"])}
f_w = (1 - d["chi_wave_mitig"]) / (1 - ew)
print("window retention factors (wave arm):", np.round(f_w[25:43], 3))
for thr in (0.01, 0.15, 0.2, 0.25, 0.3, 0.35):
    def odr_thr(chi, chi_cal, chi_exact, L, suppression_threshold=thr):
        return R.odr_mitigate(chi, chi_cal, chi_exact, L, suppression_threshold)
    hr = dict(d); hr["X_mit"] = odr_thr(d["chi_wave"], d["chi_wave_mitig"], ew, 34) - odr_thr(d["chi_vacuum"], d["chi_vacuum_mitig"], evac, 34)
    hr["sigma"] = np.full(68, 0.03)
    n_nan = int(np.sum(~np.isfinite(hr["X_mit"][25:43])))
    run(f"ex4.2 default suppression_threshold={thr} ({n_nan} NaN in W)", ff.grade_ex4_2, hr, job, odr_thr)
