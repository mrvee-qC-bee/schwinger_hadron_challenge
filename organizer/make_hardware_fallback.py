"""
Organizer tool: build the two fallback datasets used by Part 4 of the notebook when
RUN_ON_HARDWARE = False (or when a team's job did not return).

    python organizer/make_hardware_fallback.py

Writes
    organizer/fallback_data/hardware_fallback.npz   (main t=8 run, 4 PUBs)
    organizer/fallback_data/canary_fallback.npz     (t=0 canary, 2 PUBs)

Sources
    reference_data/hardware_ibm_kingston_2026-07-25.npz + _provenance.json
        keys odr__T8__z_{wave,mitig_wave,vacuum,mitig_vacuum}[_std]  (raw <Z_j>, 68 sites)
        -> evs = (-1)^j <Z_j> + 1  (chiral-condensate observables), stds unchanged
    organizer/fallback_data/canary_ibm_boston_2026-07-27.json
        measured.t0_wave / measured.t0_vacuum are <Z_j> (20000 shots, no twirling stds recorded)
        -> evs = (-1)^j <Z_j> + 1, stds = binomial shot noise sqrt((1 - <Z>^2)/shots)

Usage is not recorded in either source; it is ESTIMATED with the QPU-plan model
2 s + 0.45 ms x executions and flagged as such (usage_estimated=True).
Release policy: copy the two files to reference_data/ on Day-2 morning for teams without a job.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RD = os.path.join(ROOT, "reference_data")
OUT = os.path.join(HERE, "fallback_data")
USAGE_PER_EXEC = 0.45e-3   # s, QPU-plan model
USAGE_OVERHEAD = 2.0       # s per job


def z_to_chi(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    j = np.arange(z.size)
    return (-1.0) ** j * z + 1.0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # ---------------- main run (ibm_kingston, ODR strategy, T=8) ----------------
    src = os.path.join(RD, "hardware_ibm_kingston_2026-07-25.npz")
    prov = json.load(open(os.path.join(RD, "hardware_ibm_kingston_2026-07-25_provenance.json")))
    d = np.load(src)
    order = ["wave", "mitig_wave", "vacuum", "mitig_vacuum"]   # -> [wave, wave_mitig, vacuum, vacuum_mitig]
    evs = np.stack([z_to_chi(d[f"odr__T8__z_{k}"]) for k in order])
    stds = np.stack([np.asarray(d[f"odr__T8__z_{k}_std"], dtype=float) for k in order])
    layout = np.asarray(d["initial_layout"], dtype=int)
    meta = prov["odr__T8"]
    tw = meta["options"]["twirling"]
    n_rand, shots = int(tw["num_randomizations"]), int(tw["shots_per_randomization"])
    usage = USAGE_OVERHEAD + USAGE_PER_EXEC * n_rand * shots * 4
    np.savez(
        os.path.join(OUT, "hardware_fallback.npz"),
        evs=evs, stds=stds, layout=layout,
        job_id=str(meta["job_id"]), backend=str(meta["backend"]), submitted=str(meta["submitted"]),
        usage_s=float(usage), usage_estimated=True,
        num_randomizations=n_rand, shots_per_randomization=shots,
        dd="XY4", resilience_level=0, t=8.0, fallback=True,
        note="ibm_kingston 2026-07-25, strategy 'odr' (gate+measure twirling, DD XY4, resilience 0); "
             "evs are chi_j = (-1)^j Z_j + 1 for [wave, wave_mitig, vacuum, vacuum_mitig]; usage estimated.",
    )
    print(f"hardware_fallback.npz: evs {evs.shape}, stds {stds.shape}, layout {layout.shape}, "
          f"job {meta['job_id']} on {meta['backend']}, {n_rand} x {shots} shots, usage ~{usage:.0f} s (estimated)")

    # ---------------- canary (ibm_boston, t=0, 20000 shots) ----------------
    c = json.load(open(os.path.join(OUT, "canary_ibm_boston_2026-07-27.json")))
    zw = np.asarray(c["measured"]["t0_wave"], dtype=float)
    zv = np.asarray(c["measured"]["t0_vacuum"], dtype=float)
    shots_c = int(c["shots"])
    evs_c = np.stack([z_to_chi(zw), z_to_chi(zv)])
    stds_c = np.stack([np.sqrt(np.clip(1 - zw**2, 0, None) / shots_c),
                       np.sqrt(np.clip(1 - zv**2, 0, None) / shots_c)])
    usage_c = USAGE_OVERHEAD + USAGE_PER_EXEC * shots_c * 2
    np.savez(
        os.path.join(OUT, "canary_fallback.npz"),
        evs=evs_c, stds=stds_c, layout=np.asarray(c["layout"], dtype=int),
        job_id=str(c["job_id"]), backend="ibm_boston", submitted=str(c.get("scored", "2026-07-27")),
        shots=shots_c, usage_s=float(usage_c), usage_estimated=True, fallback=True,
        verdict_organizer=str(c.get("verdict", "")),
        note="ibm_boston 2026-07-27 t=0 canary (20000 shots, no twirling); evs are chi_j for "
             "[wave_init, vacuum_init]; stds = binomial shot noise; usage estimated.",
    )
    print(f"canary_fallback.npz: evs {evs_c.shape}, job {c['job_id']} on ibm_boston, {shots_c} shots, "
          f"usage ~{usage_c:.0f} s (estimated), organizer verdict '{c.get('verdict')}'")


if __name__ == "__main__":
    main()
