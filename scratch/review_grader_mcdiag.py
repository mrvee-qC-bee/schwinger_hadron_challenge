"""Diagnose the ex3.1 odr_uncertainty seed lottery: per-site relative deviation of a correct 2000-sample MC (various seeds)
from the grader's 10^4-sample MC."""
from review_grader_common import setup, ROOT
import numpy as np
ff = setup("mcdiag")
import schwinger_reference as R
L = 8
chi_true, chi_exact, f, chi_cal, chi_meas = ff._synthetic_odr(L, 3200)
rng = np.random.default_rng(3201); chi_std = rng.uniform(0.01, 0.05, 2 * L); cal_std = rng.uniform(0.01, 0.05, 2 * L)
def mc(n, seed):
    rg = np.random.default_rng(seed); S = np.empty((n, 2 * L))
    for s in range(n):
        S[s] = ff._qdc_odr(chi_meas + chi_std * rg.normal(size=2 * L), chi_cal + cal_std * rg.normal(size=2 * L), chi_exact, L, 0.01)
    return S
ref = mc(10000, 3201)   # NOT the grader's exact stream (grader continues rng 3201 after drawing the stds) -- close enough for a diagnosis
grader_rng = np.random.default_rng(3201); grader_rng.uniform(0.01, 0.05, 2 * L); grader_rng.uniform(0.01, 0.05, 2 * L)
G = np.empty((10000, 2 * L))
for s in range(10000):
    G[s] = ff._qdc_odr(chi_meas + chi_std * grader_rng.normal(size=2 * L), chi_cal + cal_std * grader_rng.normal(size=2 * L), chi_exact, L, 0.01)
mc_std = np.nanstd(G, axis=0)
mask = (f > 0.2) & np.isfinite(mc_std) & (mc_std > 0)
print("site  f      pooled-f  cal_std/(1-ex)/f  grader MC std   MAD-std   max|sample|")
for j in range(2 * L):
    mad = 1.4826 * np.nanmedian(np.abs(G[:, j] - np.nanmedian(G[:, j])))
    print(f"{j:3d}  {f[j]:.3f}   {'*' if mask[j] else ' '}      {cal_std[j]/(1-chi_exact[j])/f[j]:.2f}          {mc_std[j]:.4f}      {mad:.4f}    {np.nanmax(np.abs(G[:, j])):.1f}")
print("\nworst relative deviation of a correct 2000-sample MC vs the grader's 10^4 MC, per seed:")
for seed in range(30):
    S = mc(2000, seed); sd = np.nanstd(S, axis=0)
    rel = np.abs(sd[mask] - mc_std[mask]) / mc_std[mask]
    j = np.arange(2 * L)[mask][np.argmax(rel)]
    print(f"  seed {seed:2d}: worst {np.max(rel):.2f} at site {j} (f={f[j]:.3f})", "FAIL" if np.max(rel) >= 0.25 else "")
print("\n10^4-sample MC vs 10^4-sample MC with another seed (the reference itself):")
for seed in (1, 2, 3):
    sd = np.nanstd(mc(10000, seed), axis=0); rel = np.abs(sd[mask] - mc_std[mask]) / mc_std[mask]
    print(f"  seed {seed}: worst {np.max(rel):.2f} at site {np.arange(2*L)[mask][np.argmax(rel)]}")
