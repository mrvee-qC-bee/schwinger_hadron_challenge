"""
Package the participant and organizer kits.

    python tools/make_kits.py            # writes ../schwinger-hadron-challenge-{participant,organizer}.zip

Participant kit  = the challenge folder minus organizer/, tools/, scratch/, submission/, __pycache__,
                   the solution notebook, and the organizer-only reference data (the cached
                   ibm_kingston set and the bond-128 MPS columns are released later, with the
                   fallback dataset, only to teams that could not run their own job).
Organizer kit    = everything except scratch/, submission/ and __pycache__.
"""
from __future__ import annotations

import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.dirname(ROOT)
NAME = os.path.basename(ROOT)

SKIP_DIRS_ALWAYS = {"scratch", "submission", "__pycache__", ".ipynb_checkpoints"}
SKIP_DIRS_PARTICIPANT = SKIP_DIRS_ALWAYS | {"organizer", "tools"}
SKIP_FILES_PARTICIPANT = {
    "schwinger_hadron_solution.ipynb",
    # organizer-only reference data: raw hardware from the July-2026 campaign (released with the
    # fallback dataset on Day 2) -- the participant notebook degrades gracefully without them
    os.path.join("reference_data", "hardware_ibm_kingston_2026-07-25.npz"),
    os.path.join("reference_data", "hardware_ibm_kingston_2026-07-25_provenance.json"),
    os.path.join("reference_data", "make_mps_reference.log"),
    # Bonus B1's answer: the dt = 0.5 / 0.25 profiles the exercise asks the team to produce.
    # The notebook loads it only "if released" and computes its own otherwise.
    os.path.join("reference_data", "mps_reference_L34_dt.npz"),
}


def collect(participant: bool) -> list[str]:
    skip_dirs = SKIP_DIRS_PARTICIPANT if participant else SKIP_DIRS_ALWAYS
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            if participant and rel in SKIP_FILES_PARTICIPANT:
                continue
            files.append(rel)
    return files


def write_zip(participant: bool) -> str:
    tag = "participant" if participant else "organizer"
    path = os.path.join(OUT_DIR, f"{NAME}-{tag}.zip")
    files = collect(participant)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(os.path.join(ROOT, rel), os.path.join(NAME, rel))
    size = os.path.getsize(path) / 1e6
    print(f"\n{path}  ({len(files)} files, {size:.1f} MB)")
    for rel in files:
        print("   ", rel)
    return path


if __name__ == "__main__":
    p = write_zip(participant=True)
    o = write_zip(participant=False)
    # a participant kit that still contains an answer would be a release bug
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
    bad = [n for n in names if "solution" in n.lower() or "/organizer/" in n or "/tools/" in n]
    if bad:
        print("\nERROR: participant kit contains organizer material:", bad)
        sys.exit(1)
    print("\nparticipant kit checked: no solution notebook, no organizer/ or tools/ material")
