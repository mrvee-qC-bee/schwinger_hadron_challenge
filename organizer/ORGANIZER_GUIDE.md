# Organizer guide — "Hadron Dynamics in the Schwinger Model: Trust, but Verify"

UofT Qiskit Fall Fest 2026 edition. This file is for organizers, mentors and judges only; it
contains answers. The contract for everything below is `organizer/SPEC.md`; the source of truth for
every number is `organizer/schwinger_reference.py` (verified by `tools/test_reference.py`).

Contents: 1 pitch · 2 rubric · 3 timelines and hardware windows · 4 accounts and QPU plan ·
5 grading procedure · 6 judge sheet and viva bank · 7 hint policy · 8 pitfalls · 9 regeneration ·
10 provenance and licensing · 11 answer key at a glance · 12 known deviations from SPEC.md

---

## 1. Pitch

Teams reproduce the central result of *Quantum Simulations of Hadron Dynamics in the Schwinger Model
using 112 Qubits* (arXiv:2401.08044) at reduced scale — a hadron wavepacket on 68 qubits (L = 34
staggered sites) evolved to t = 8 with 8 second-order Trotter steps, ≈ 4,960 CZ gates after
transpilation — on an IBM Heron r2 device under the Open Plan, and they have to *prove* every layer of
the pipeline before they spend a second of QPU time: the Hamiltonian against exact diagonalization,
the electric-field circuit as an exact unitary identity, the pre-trained SC-ADAPT-VQE vacuum
against the true ground state, the Trotter error by a dt-ratio test, the classical MPS reference by
bond-dimension convergence, the layout by calibration-aware path search, and their own
operator-decoherence-renormalization (ODR) code on synthetic noise. The scored target is the
vacuum-subtracted chiral condensate profile X_j at t = 8 in the window j = 25…42: two side peaks
(X ≈ 0.50–0.52 at sites 31, 32, 35, 36) around a central dip (X ≈ −0.006 at 33, 34).

**Why this is harder than Fall Fest 2025.** The 2025 kit shipped two 1–5-qubit labs (Bell states,
small VQE) with a server grader. This kit is a utility-scale Hamiltonian-simulation experiment:
68 qubits, 2q-depth ≈ 220, heavy-hex layout on a 156-qubit chip, runtime `EstimatorV2` twirling and
dynamical decoupling, a hand-written mitigation method with a bias derivation, an explicit QPU
budget the teams must plan and defend, and a 10-point viva. Everything still runs on a laptop
(Aer MPS reproduces the reference in seconds; `FakeKingston` for noise), so the barrier is
understanding, not compute.

**How it differs from the public QDC 2025 notebook, and why copying does not help.** The base
notebook (`qiskit-community/qdc-challenges-2025`, Track B lab 2) had 12 server-graded fill-ins, and a
fully solved fork by a member of a winning Track-B team is public. The QDC grader endpoint (`qc_grader.challenges.qdc_2025`) was
removed in April 2026, so that notebook cannot even be graded any more. The design neutralises the
public solution as follows:

* The QDC fill-ins are only a 10-point warm-up (Part 0) and are graded by *calling the team's
  functions at other lattice sizes* (L = 6, 8, random angles) against stored statevector fingerprints.
* 90 of 100 points are for exercises that do not exist in the QDC notebook: Part 1 (physics checks,
  20 pts), Part 2 (engineering the 68-qubit run, 15), Part 3 (mitigation written by the team, 20),
  Part 4 (hardware with a hard 180-s QPU cap, 25), Part 5 (error budget + viva, 10).
* The QDC helper `postselection_and_mitigation` is *given* in `challenge_utils.py`, but ex 3.1 requires
  a team-written `odr_mitigate`, an uncertainty propagation and a bias formula, and the hardware
  score in ex 4.2 is recomputed from the raw evs — locally with the team's own `odr_mitigate`, in the
  organizer re-grade with the `odr_mitigate` the notebook exports to `team_functions.py` (or, failing
  that, with the QDC formula that ex 3.1 forces it to reproduce).
* The Trotter ordering is pinned to Fig. 8 of the paper (odd bonds (1,2),(3,4),… first in the
  opening kinetic half-step, even bonds first in the closing one). This is the ordering behind the
  QDC bond-40 files and the cached Kingston run (§12), so a copied QDC solution *passes* ex 0.5 — it
  is only worth 2 points — while the 90 non-QDC points and the viva remain.
* The viva (Part 5) asks for the reasoning behind the numbers; the 20-question bank in §6 is designed
  so that someone who ran a copied notebook cannot answer it.

---

## 2. Rubric (100 + 6 bonus)

| Part | id | name | pts | scored by |
|---|---|---|---|---|
| 0 Warm-up (QDC fill-ins, function-level) | 0.1 | `chiral_condensate_observables(L)` | 2 | auto |
| | 0.2 | `RXYplus(theta)`, `RXYminus(theta)` | 2 | auto |
| | 0.3 | `prep_strong_coupling_vacuum`, `vacuum_prep_rotate_OV_3`, `prep_vacuum` | 2 | auto |
| | 0.4 | `wave_prep_rotate_O_22`, `prep_wave` | 2 | auto |
| | 0.5 | `trotter_step`, `evolve_circuits` | 2 | auto |
| 1 Physics you can verify | 1.1 | `schwinger_hamiltonian`, `electric_hamiltonian_truncated`, charge-sector checks, `E0_L8` | 5 | auto |
| | 1.2 | `electric_layer(L,t,g)` == exp(−i t H_el^(1)) + `truncation_shift` | 4 | auto |
| | 1.3 | `vqe_fidelity`, `vqe_energy_gap` at L = 4, 6, 8 | 3 | auto |
| | 1.4 | Trotter error table L = 8, ratios, Richardson, `dominant_error` | 5 | auto |
| | 1.5 | MPS bond-dimension convergence L = 34, t = 8; bond-20 artefact | 3 | auto |
| 2 Engineering the 68-qubit experiment | 2.1 | 4 ISA circuits + observables on one layout | 3 | auto |
| | 2.2 | Gate accounting + noise-matched calibration circuit (`evolve_circuits_matched`) | 5 | auto |
| | 2.3 | `select_chain(backend, n_qubits=68)` calibration-aware path + re-transpile | 5 | auto |
| | 2.4 | `flight_plan` + `make_estimator_options(backend)` + usage prediction | 2 | auto |
| 3 Mitigation you wrote yourself | 3.1 | `odr_mitigate`, `odr_uncertainty`, `odr_bias` on synthetic data | 7 | auto |
| | 3.2 | `mitigation_options`: Runtime `EstimatorOptions` (twirling, DD, resilience) of every hardware job | 5 | auto |
| | 3.3 | Noisy rehearsal L = 6, t = 4 with `reduced_noise_model`, own ODR, RMSE vs exact | 8 | auto |
| 4 Hardware | 4.1 | Canary at t = 0 (2 PUBs × 16 twirls × 128 shots, ≤ 30 s usage) | 3 | auto |
| | 4.2 | Main t = 8 run (4 PUBs × 64 twirls × 256 shots) → leaderboard metric | 18 | auto (+ judge flags) |
| | 4.3 | Improvement run (≤ 60 s usage) with z-score | 4 | auto |
| 5 Error budget + defence | 5 | one-page error-budget table, ODR bias derivation, 8-min pitch, viva | 10 | **judges** (§6) |
| Bonus (max +6, outside the 100) | B1 | L = 34 dt→0 Richardson (dt = 0.5, 0.25 at bond 32) | 3 | auto (structural) + organizer arrays |
| | B2 | ODR on amplitude-damping vs depolarizing toy at L = 4 (untwirled); charge-witness ordering | 2 | auto |
| | B3 | Fractional-gate barbell (`rzz`) unitary check | 1 | auto |

"Auto" = `fallfest_grader.py` locally during the event, re-run by `organizer/grade_submission.py`
on the submitted folder. Ex 4.2 is automatic but carries two judge-controlled items: the
"quiet-region" 2 points can be granted on a written systematics note even when
`quiet > 0.05`, and any integrity flag (§5) zeroes it pending judge review.

Hardware metric (ex 4.2) for reference — window W = {25,…,42}, X_ref = bond-64 MPS:

```
RMSE_W   = sqrt(mean_{j∈W} (X_hat_j − X_ref_j)^2)       (an unreported / NaN window site is charged an
           error of max(|X_ref_j|, 0.25) for both points and rank, so hiding a site never helps; a NaN outside W
           counts as 1.0 in the quiet median; sites 31–36 must all be finite for the contrast and shape
           points; fewer than 16 finite window sites → NOT RANKED)
C        = mean(X_hat[31,32,35,36]) − mean(X_hat[33,34]),  C_ref = 0.514 (measured from the bond-64
           reference; SPEC's rounded constant is 0.515)
quiet    = median_{j∉W} |X_hat_j|
points   = 10·clip(1 − RMSE_W/0.25) + 4·clip(1 − |C − C_ref|/0.35)
         + 2·[C ≥ 0.20 and min(X_hat[31,32,35,36]) − max(X_hat[33,34]) ≥ 0.25]
         + 2·[quiet ≤ 0.05 or judge-accepted systematics note]
gates    : 4 PUBs on one layout; usage ≤ 180 s (else −5); ODR factors < 0.9 on ≥ 10 window sites;
           fallback=True entries capped at 75 % and flagged.
```

Calibration with the cached July-2026 ibm_kingston ODR set (100 twirls × 1000 shots, more shots
than teams get), scored against the pinned-ordering bond-64 reference: RMSE_W 0.125, C 0.437,
quiet 0.021 → 5.0 + 3.1 + 2 + 2 ≈ 12/18 with the formula as written (an earlier SPEC draft quoted
≈ 10/18 assuming the quiet term is lost; SPEC now says ≈ 12/18, see §12). Raw twirl+DD without ODR: RMSE_W 0.229, C 0.068 → ≈ 1/18. A team at
64 × 256 shots that reaches RMSE_W ≈ 0.15 is doing well; RMSE_W < 0.10 is a leaderboard-topping
result (the cached TREX + ODR combination reaches 0.052).

---

## 3. Timelines and hardware windows

### Before the event (checklist)

1. Classroom Account requested at least two weeks ahead; the promotion opt-in announced (§4).
2. Kit verified on the organizer laptop (§9): `python tools/test_reference.py`,
   `python tools/test_grader.py`, `MPLBACKEND=Agg python tools/smoke_part_a.py` and `smoke_part_b.py`.
3. Participant `README.md` placeholders filled in: the kit URL under *Install*, the *Timeline* table,
   the deadline and drop-off location under *Submission format*.
4. Level-2 hint texts (§7) ready to post on Day 1 at 18:00; the hardware windows below announced.
5. Fallback files present in `organizer/fallback_data/` (`hardware_fallback.npz`, `canary_fallback.npz`;
   regenerate with `python organizer/make_hardware_fallback.py`, §4).
6. Judge sheets printed from `organizer/judge_sheet.md` (one per team and judge).

### 3-day schedule (recommended)

| When | What | Notes |
|---|---|---|
| **Day 0 (−14 d)** | Classroom Account request submitted; instance per participant | §4 |
| **Day 0 (−7 d)** | Kit released: `README.md`, `requirements.txt`, participant notebook; env check `ff.check_env()` | teams install before arriving |
| **Day 1 09:00** | Kick-off: physics primer (30 min), rules, QPU cap, submission format, viva format | mention that the public QDC solution earns ≤ 10 pts |
| Day 1 09:45–13:00 | Parts 0–1 (warm-up + physics checks) | mentors circulate; level-1 hints are inline |
| Day 1 13:00–16:00 | Part 2 (transpile, gate accounting, `select_chain`, flight plan) | organizer verifies each team's `flight_plan` usage ≤ 180 s before any submission |
| **Day 1 15:00–17:00** | **Canary window** (ex 4.1, t = 0, ≈ 4 s usage) | first real job; teams read `job.usage()` and record the job id |
| Day 1 17:00–19:00 | Part 3.1–3.2 (ODR + Runtime options) | level-2 hints for 2.2 and 3.1 released at 18:00 (§7) |
| **Day 1 19:00–21:00** | **Main-run window** (ex 4.2, ≈ 32 s usage) | submit before leaving; the queue clears overnight |
| Day 2 09:00 | Fallback dataset released to teams *without a returned job* (§4) | capped at 75 % |
| Day 2 09:00–12:00 | Part 3.3 (noisy rehearsal) + post-processing of the main run | |
| **Day 2 12:00–14:00** | **Improvement window** (ex 4.3, ≤ 60 s usage) | teams must show the z-score plan to a mentor first |
| **Day 2 15:00** | **Hardware freeze** — no further jobs count | on Day 3 morning check `submitted` in each `job_info.json` by hand (§5); jobs after 15:00 do not count |
| Day 2 15:00–18:00 | Part 5 error budget, report, bonus | |
| Day 3 09:00–11:00 | Submission deadline 09:00; organizer grading (§5) | ≈ 3 min per team automated |
| Day 3 11:00–15:00 | 8-min pitch + 7-min viva per team (§6) | 3 judges, 15-min slots, 12 teams ≈ 3 h |
| Day 3 15:30 | Leaderboard + awards | rank by RMSE_W, ties by full-68-site RMSE, then lower usage |

### 2-day schedule

Drop the bonus, ex 4.3 (improvement run) and make ex 1.5 optional (MPS bond-64 arrays are
loaded, not recomputed). Rescale: Part 4 becomes 21 pts (4.1 = 3, 4.2 = 18), Part 1 becomes 17 pts
(1.5 = 0, add its 3 pts to 1.4 as a second Richardson entry, or simply grade out of 97 and scale).
Compressed windows: canary Day 1 14:00–16:00, main run Day 1 17:00–20:00, fallback release
Day 2 09:00, freeze Day 2 11:00, submission Day 2 14:00, viva Day 2 14:30–17:00.

### Hardware windows — rationale

* Canary on Day 1 afternoon: proves credentials, layout and `job.usage()` before the expensive run.
* Main run Day 1 evening: Open-Plan queues on the three Heron devices are typically hours long
  during North-American daytime; overnight jobs return by Day 2 morning.
* Improvement run Day 2 midday, freeze Day 2 afternoon: leaves an afternoon for post-processing and
  the report, and makes the leaderboard final before the viva.

---

## 4. Accounts and QPU plan

### Plans (docs checked 2026-09-07)

* **Open Plan**: 10 QPU-minutes per 28-day rolling window per instance, us-east region only,
  **job and batch execution modes only (no sessions)**, on `ibm_fez`, `ibm_marrakesh`,
  `ibm_kingston` (Heron r2, 156 qubits). <https://quantum.cloud.ibm.com/docs/guides/plans-overview>
* **Promotion**: since 16 March 2026, *active* Open-Plan users can opt in to +180 minutes over
  12 months (IBM blog "Doubling down on open-access quantum computing"). Ask participants to opt in
  the week before the event; it is per-user, not automatic.
* **Classroom Account** (recommended): an educator requests one at <https://ibm.biz/classroom-account>
  **at least two weeks ahead**. Approval e-mail carries a 32-digit feature code → create a *new*
  IBM Cloud account with it (no credit card; institution in the first-name field, course in the
  last-name field) → copy the Account ID from IBM Quantum Platform → send it back to IBM to activate.
  Then Access management → Invite user (multiple e-mails at once; one Open-Plan instance per user is
  created automatically, prefix e.g. `fallfest26`). Each student instance has its own 10 min.
  <https://quantum.cloud.ibm.com/docs/guides/classroom-accounts>
* Qiskit Functions (Q-CTRL, Algorithmiq TEM, Qedma) need Premium — out of scope; say so on Day 1.

### Per-team budget (hard cap 180 s of usage)

Usage model used by the notebook's `flight_plan` preflight:

```
usage_s ≈ 2 s + 0.45 ms × executions,   executions = Σ_PUBs (num_randomizations × shots_per_randomization)
```

IBM's documented quick formula is `2 + 0.00035 × executions`
(<https://quantum.cloud.ibm.com/docs/guides/estimate-job-run-time>); 0.45 ms is that number scaled to the
observed 6-minute QDC run (4 × 480 × 400 = 768,000 executions) so the preflight is conservative.

| job | PUBs × twirls × shots | executions | predicted | measured (cached) |
|---|---|---|---|---|
| canary (ex 4.1, t = 0) | 2 × 16 × 128 | 4,096 | ≈ 4 s | ibm_boston canary, 20,000 shots ≈ 11 s |
| main (ex 4.2, t = 8) | 4 × 64 × 256 | 65,536 | ≈ 32 s | Kingston ODR set 4 × 100 × 1000 ≈ 180 s |
| improvement (ex 4.3) | free, ≤ 60 s | ≤ 130,000 | ≤ 60 s | |
| **total** | | | **≈ 96 s** | cap 180 s, Open Plan 600 s |

Enforcement: (1) the notebook refuses to submit unless `flight_plan["predicted_usage_s"] ≤ 180`
and `estimator.options.max_execution_time` is set (180 s); (2) after each job the team records
`job.usage()` in `job_info["usage_s"]`; (3) the grader subtracts 5 points from ex 4.2 when the recorded
`usage_s` exceeds 180 s, and on Day 3 morning an organizer re-fetches `job.usage()` for every job id by
hand (§5) and corrects `usage_s` if a team under-reported it.
Teams may not split the four PUBs of one run across accounts/instances (the four circuits must share
one layout *and one calibration state*; the grader checks one `job_id` per run).

### If the queue is long

* Check <https://quantum.cloud.ibm.com/computers> for pending jobs; switch the whole run to the
  Heron with the shortest queue (`select_chain` re-runs in < 1 min on any of the three).
* Never cancel a queued job to "resubmit somewhere faster": a canceled job that had started counts.
* Do not open sessions (not available on the Open Plan; a session would also burn wall-clock time).
* If a team's main run has not returned by Day 2 09:00, release the fallback dataset (below); if it
  returns later, the team may replace the fallback entry before the freeze.

### Fallback dataset release

1. The files are pre-built in `organizer/fallback_data/`: `hardware_fallback.npz` (the T = 8 ODR set of
   the July-2026 ibm_kingston campaign: evs/stds, layout, job id, `fallback=True`) and
   `canary_fallback.npz` (the ibm_boston t = 0 canary). Rebuild them with
   `python organizer/make_hardware_fallback.py`, which reads
   `reference_data/hardware_ibm_kingston_2026-07-25.npz` and `organizer/fallback_data/canary_ibm_boston_2026-07-27.json`.
2. At Day 2 09:00 copy into the `reference_data/` folder of each team **without a returned main job**:
   `hardware_fallback.npz` (ex 4.2) and, if they had no canary either, `canary_fallback.npz` (ex 4.1).
   Teams that want the 4.3 *illustration* (cached strategies compared by z-score) additionally need
   `reference_data/hardware_ibm_kingston_2026-07-25.npz` + `_provenance.json` (organizer kit only).
   With `RUN_ON_HARDWARE = False` the notebook picks the files up automatically (`load_fallback`).
3. Their ex 4.1 / 4.2 scores are capped at 75 % (2.25/3, 13.5/18) and the leaderboard line is flagged
   `fallback`. They still must run *their own* `odr_mitigate` on it (the grader recomputes X_mit).

---

## 5. Grading procedure

**Submission** (deadline Day 3 09:00, or Day 2 14:00 in the 2-day plan): one zip named
`<team>.zip` containing

```
submission/            score.json, ex*.npz written by fallfest_grader (do not edit)
                       circuits_isa.qpy, layout.json, flight_plan.json (2.1–2.4), mitigation_options.json (3.2), rehearsal_L6.npz (3.3)
                       canary_t0.npz, canary_job_id.txt, hardware_t8.npz, job_id.txt,
                       improvement.npz, job_info.json, error_budget.md, report.md (+ figures)
                       team_functions.py   (the notebook's last cells export the team's functions)
schwinger_hadron_participant.ipynb     executed, outputs included
```

**Run the organizer grader** (the environment built from `requirements.txt`; fully offline — the
live job checks are done by hand, see "What judges re-fetch live" below):

```
python organizer/grade_submission.py submissions/<team>.zip                      # table + LEADERBOARD line
python organizer/grade_submission.py submissions/<team>/submission --hidden     # unzipped folder, hidden mode
python organizer/grade_submission.py submissions/*.zip --leaderboard leaderboard.csv   # all teams, ranked CSV
```

It accepts a `<team>.zip`, the unzipped team folder or its `submission/` folder, writes
`organizer_score.json` next to `score.json` (for a zip: `<team>.organizer_score.json` next to the zip)
and, with `--leaderboard`, a CSV sorted the way the leaderboard is ranked (≈ 10 s per team;
`--hidden` adds two `select_chain` runs, ≈ 40 s).

* It re-scores every **array-graded** exercise from `submission/exN.npz` (1.3–1.5, 2.1 layout, 2.2
  counts, 2.3 chain validity and cost, 2.4, 3.3 arrays, 4.1, 4.2) and recomputes the hardware metric
  from the raw evs.
* **Function-level** exercises (0.1–0.5, 1.1, 1.2, 3.1, 3.2) are re-run from `submission/team_functions.py`,
  which the notebook's last code cells write (every function of the notebook plus its constants; the
  cell test-imports the file). Without that file their locally reported points are kept
  **provisionally** and the tool prints a `NOT RE-VERIFIED` line naming them: check those exercises at
  the viva (ask the team to run the grader cell in front of you).
* **Tamper check**: every entry of `score.json` is compared with the last `[exN] p/m` grader line
  printed in the executed notebook; a score the notebook never printed, or a different number, is
  flagged `TAMPER?`. (A team that re-ran a grader cell after executing the notebook produces the same
  mismatch — ask them to re-execute; the flag is a prompt for a question, not a verdict.)
* **Hidden mode** (`--hidden`) re-runs the team's `select_chain` on two other frozen Heron snapshots
  (FakeFez, FakeMarrakesh) and averages the ex 2.3 points; it needs `team_functions.py`. The public and
  hidden graders otherwise use the same references: the defence against hard-coding is the function
  re-run at lattice sizes the team did not tune for (L = 5, 7 and 6, 8) plus the viva, not a second
  seed set.
* **Leaderboard**: rank by ex 4.2 `RANK-RMSE` (the window RMSE with unreported sites charged max(|X_ref_j|, 0.25),
  printed by `grade_submission.py`), ties by RMSE over all 68 sites, then by lower total usage.
  A submission with fewer than 16 of the 18 window sites reported is listed "NOT RANKED".
  `fallback` rows are listed but ranked below every genuine row.
* **The answer key is sealed, not secret.** Since 2026-09-10 `reference_data/grader_refs.npz` keeps
  the oracle (statevectors, Hamiltonian coefficients, energies, the Trotter table, gate counts) in an
  obfuscated blob rather than in clear numpy arrays, so it cannot be read with a one-line `np.load`.
  This is deliberately *not* cryptography — the grader has to carry its own oracle to a participant
  laptop, and anyone who reads the grader source can print every key with `ff._refs()[key]`; the blob
  only stops accidental discovery through `np.load`. Treat reading it exactly like editing
  `submission/`: a disqualification, detectable in the viva (a team that "knows" E0 or the Trotter
  table without being able to derive it). The checks that actually carry weight are the function
  re-runs from `team_functions.py`, the array re-grade, the tamper check and the defence.
* **Integrity checks — automatic** (failure → 0 points on the affected exercise and a judge flag):
  1. `X_hat` recomputed from the submitted 4 PUB evs with the team's `odr_mitigate` (or the QDC
     formula) must equal the submitted `X_mit` to 1e-6.
  2. All four PUB circuits share one `initial_layout` (68 distinct physical qubits, a connected path on
     the frozen coupling map); the observables' `apply_layout` consistency is checked locally only.
  3. `score.json` agrees with the grader lines in the executed notebook (tamper check above).
* **Integrity checks — by hand** (Day 3 morning, one organizer with the Classroom-Account admin login;
  a failure is entered as a judge flag and the exercise zeroed by the organizer):
  4. `usage_s` per job matches `job.usage()` within 10 %; team total ≤ 180 s (else −5 on ex 4.2).
  5. Job timestamps (`job_info.json` → `submitted`) lie inside the hardware windows and before the freeze.
  6. The submitted evs are not identical (to 1e-9 on any site) to an array in the organizer-released
     `hardware_ibm_kingston_2026-07-25.npz` or the fallback files unless `fallback=True` is declared:
     `np.load` both and compare the four `chi_*` rows of `submission/ex4_2.npz` (the Kingston file
     contains *all* strategies — baseline, twirl_dd, odr, trex, zne_fold, … — so this matters).
* **What judges re-fetch live** (one organizer laptop with a Classroom-Account admin login, Day 3
  morning): for each job id in `job_info.json`, `job.usage()`, `job.backend().name`,
  `job.metrics()["timestamps"]`, the number of PUBs and `job.inputs["options"]` (to confirm twirling
  was on for the ODR run and that no session was used). Ten seconds per team; no QPU time.

---

## 6. Judge sheet — Part 5 (10 points) and the viva bank

Three judges, one sheet per team, 15-minute slot (8-min pitch, 7-min viva). Score independently,
average, round to 0.5. A printable sheet is in `organizer/judge_sheet.md` (one per team and judge);
the judges choose which team member answers each of the 8 questions.

| item | pts | 0 | half | full |
|---|---|---|---|---|
| Error-budget completeness | 3 | list of buzzwords | table with ≥ 4 of the 7 rows below but no propagation to X | all 7 rows, each with a mechanism, a sign/shape and how it was measured or bounded |
| Magnitudes | 3 | none or wrong by > 10× | right order of magnitude for ≥ 4 rows | numbers consistent with their own notebook (Trotter dt = 1 error, MPS bond gap, shot noise from stds, retention factors, ODR bias) |
| Viva | 4 | cannot explain their own code | answers 4–6 of 8 questions | answers ≥ 7 of 8, including one "why" question |

The seven error-budget rows: (1) Trotter dt = 1 (state-dependent, does **not** cancel in the
vacuum subtraction), (2) electric-field range-1 truncation, (3) SC-ADAPT-VQE vacuum infidelity,
(4) MPS bond truncation of the *reference*, (5) shot noise + twirl-sampling noise, (6) decoherence
retention (what ODR divides out) and its residual non-Pauli part, (7) ODR ratio bias and
retention threshold/mirror averaging. Expected magnitudes in §11.

### Viva bank (ask 8, mix of ★ recall and ★★ reasoning)

1. ★ *Why does the Hamiltonian carry a `+ m/2 · I` per site?* — So that H_m = m Σ_j n_j counts the
   rest mass of occupied fermion sites and the strong-coupling vacuum has zero mass energy. It is a
   constant shift m·L (= 4.0 at L = 8, m = 0.5): E0(L = 8) = −2.50901 with it, −6.50901 without. It
   changes no dynamics or observable; the grader ignores the identity coefficient.
2. ★★ *Why is the electric ("barbell") layer exact, with no Trotter error of its own?* — The truncated
   H_el^(Q=0)(1) contains only Z and ZZ terms, which all commute, so exp(−i t H_el) factorises exactly
   into Rz and ZZ rotations; the 4-qubit barbell implements those ZZ rotations. Verified as a unitary
   identity to 1e-8 at L = 4 and 6 (`tools/test_reference.py`, both tiling branches). The only Trotter
   error is between H_kin / H_el / H_m and between even and odd kinetic bonds.
3. ★ *What does the range-1 truncation of the electric term change?* — The Coulomb term
   (g²/2) Σ_j (Σ_{k≤j} Q_k)² is all-to-all; keeping charge–charge terms up to range 1 (and dropping the
   charge-0 sector constant) makes it nearest-neighbour ZZ, hence 4-qubit barbells. The shift on X at
   L = 8, t = 4 is the `truncation_shift` they measured in ex 1.2 (see §11); for m = 0.5, g = 0.3 the
   electric term is a 0.045-scale perturbation so the truncation is a second-order effect.
4. ★ *How good is the pre-trained 2-step vacuum?* — Fidelity with the exact ground state of the
   *full* H: 0.9961 (L = 4), 0.9945 (L = 6), 0.9929 (L = 8); energy gaps 0.0063 / 0.0094 / 0.0124.
   The infidelity grows roughly linearly with L (≈ 0.1 % per spatial site), so at L = 34 expect ≈ 3 %.
5. ★★ *How do you know your Trotter step is second order?* — Compare one step with exp(−i dt H_trunc)
   at L = 6: error 1.85e-3 / 2.37e-4 / 2.99e-5 for dt = 0.2 / 0.1 / 0.05 — ratios 7.8 and 8.0, i.e.
   O(dt³) local error. In ex 1.4 the same ratio test at L = 8 on X, plus Richardson extrapolation.
6. ★★ *Why does a wrong Trotter ordering still pass the ratio test, and why does the grader also check
   the ordering?* — Any symmetric (Strang) splitting has O(dt³) local error, so the ratio test passes
   for odd-first or even-first kinetic sublattices alike. But at dt = 1 the two orderings differ by
   0.023 in X at t = 8 (§12), the hardware reference (bond-64 MPS) is computed for the pinned Fig. 8
   ordering (odd-first), and the hardware score compares against it, so the ordering must match.
7. ★★ *Why does vacuum subtraction not cancel the Trotter error?* — The leading error is a
   commutator term [H_kin, H_el]-type whose expectation depends on the local state; the wavepacket
   region has different local kinetic and electric energy than the vacuum, so δchi_wave ≠ δchi_vac.
   At L = 6 the max error of chi is 0.13 (dt = 1) vs 0.06 (dt = 0.5) at t = 8; in X it shrinks but
   stays at the 0.0x level. Only *hardware* decoherence that acts multiplicatively cancels partly.
8. ★★ *Why does X converge faster than chi with the MPS bond dimension?* — Bond truncation error is
   dominated by long-range entanglement of the *vacuum* background, which is identical in the wave
   and vacuum circuits and cancels in the difference. Numbers: at t = 8, max|chi(64) − chi(128)| =
   0.015 but max|X(64) − X(128)| = 0.0007; the QDC bond-40 files differ from the bond-64 reference by
   0.019 in chi but 0.0008 in X.
9. ★ *Why is the QDC bond-20 file unusable?* — At bond 20 the truncation is so severe at t = 8 that
   the profile is not converged at all (our own bond-20 run differs from the QDC bond-20 file by 1.5
   in X, i.e. it is not even reproducible run-to-run); the bond-40 file is converged to 0.02 in X.
   Reference for scoring is bond 64 (bond 128 agrees to 0.0007).
10. ★★ *What is the mitigation circuit and why does the transpiler sabotage it?* — n/2 Trotter steps
    forward then n/2 backward: same gate structure and noise, known answer (the initial state). At the
    turning point the last forward kinetic layer R_XX(+dt/4) meets the first backward R_XX(−dt/4);
    the optimizer cancels them, so the calibration circuit has fewer CZ than the physics circuit and
    the noise no longer matches (`n_cz_physics` > `n_cz_mitig`). Fix: a barrier at the midpoint
    (`protect_midpoint=True`) or an explicitly non-cancelling junction — that is ex 2.2.
11. ★★ *Write the ODR formula and say what the retention threshold on f does.* — With chi = (−1)^j Z + 1,
    1 − chi = −(−1)^j Z, so f_j = (1 − chi_cal_j)/(1 − chi_exact_j) = ⟨Z_j⟩_cal/⟨Z_j⟩_exact is the
    retention of ⟨Z_j⟩; mitigated ⟨Z_j⟩ = ⟨Z_j⟩_meas / f_j, i.e. chi_mit = 1 − (1 − chi)/f. f and chi are
    mirror-averaged (j ↔ 2L − 1 − j) first. If f_j ≤ 0.01 the qubit has fully decohered and dividing
    would amplify noise by ≥ 100×, so the site is dropped in favour of its mirror image, or NaN if both fail.
12. ★★ *Why is twirling required for ODR?* — ODR assumes the noise channel is Pauli (diagonal Pauli
    transfer matrix) so that ⟨Z⟩ is simply scaled by a factor; coherent errors are not multiplicative
    and differ between the physics and calibration circuits. Pauli twirling converts the channel to
    Pauli noise. Cached evidence: DD-only (no twirl) + ODR gives RMSE_W 0.236 and contrast −0.10
    (worse than raw), while twirl + DD + ODR gives 0.135 / 0.42.
13. ★★ *ODR is a ratio of two noisy estimates — is it biased?* — Yes. For Z_mit = a/b with
    independent a ~ (ā, σ_a), b ~ (b̄, σ_b): E[a/b] ≈ (ā/b̄)(1 + σ_b²/b̄²), i.e. an upward bias of relative
    size (σ_b/b̄)² and variance ≈ (ā/b̄)²(σ_a²/ā² + σ_b²/b̄²). With b̄ = f·Z_exact ≈ 0.2 × 0.7 = 0.14 and
    σ_b ≈ 0.025 (64 twirls × 256 shots) the bias is ≈ 3 % and the amplified variance ≈ 18 % — that is
    the row in the error budget. The Monte-Carlo propagation in `odr_uncertainty` reproduces it.
14. ★ *What retention factors did ibm_kingston show at t = 8 and what does that imply?* — f in the
    window: 0.14–0.33, median 0.21 (twirl + DD, 5,292 CZ at O1): 70–85 % of ⟨Z⟩ is lost, the raw contrast is
    0.07 vs 0.51 ideal, and after ODR 0.44. Ideal-gate estimate: 5,292 CZ × ~0.3 % ≈ 16 errors per
    shot, consistent with retention e^{−16/68 × ...} only if errors are local — which is why per-qubit
    factors are needed rather than a global one.
15. ★★ *Why can the Open Plan not run the QDC shot counts?* — QDC used 480 twirls × 400 shots × 4
    PUBs = 768,000 executions ≈ 2 + 0.45e-3 × 768k ≈ 348 s ≈ 6 min in one job: 58 % of the whole
    10-min/28-day allowance and twice the 180-s team cap. 64 × 256 gives 65,536 executions ≈ 32 s
    with a shot-noise floor σ_Z ≈ 1/√16384 ≈ 0.008 per qubit, which ODR amplifies by 1/f ≈ 5.
16. ★ *Why one layout for all four circuits?* — ODR divides physics by calibration per physical
    qubit, and vacuum subtraction assumes the same noise on both arms; a different layout changes
    every retention factor. Also why `select_chain` must avoid dead edges (`FakeKingston` has 7).
17. ★★ *What does `select_chain` optimise?* — A 68-node simple path on the coupling map maximising
    the product of (1 − CZ error) over edges × (1 − readout error) over nodes (equivalently maximising
    the log-ESP), with dead edges (error ≥ 0.5 or None) excluded; the cached Kingston run improved the
    ESP from 1.0e-7 (default layout) to 4.8e-7. Beam/greedy search over the 156-qubit heavy-hex is
    enough; exhaustive search is not.
18. ★ *Which mitigation is yours and which is the Runtime's, and why must the ODR run be raw?* — ODR
    (3.1) is the team's own code; gate + measurement twirling, dynamical decoupling and any TREX/ZNE/PEC
    are Runtime `EstimatorOptions` produced by their `mitigation_options` (3.2). The ODR run needs
    `resilience_level = 0` and no server-side mitigation because ODR divides the physics signal by a
    calibration signal taken under the *same* noise; TREX *before* ODR (improvement run) makes the
    factors measure gate decoherence only. Local testing mode ignores every option except shots, so
    3.2 is graded on the options object. Bonus point: the charge witness Σ_j⟨Z_j⟩ = −2⟨Q⟩ = 0 exactly;
    a Pauli channel leaves it near zero, amplitude damping makes it positive (B2, untwirled).
19. ★ *What does the canary at t = 0 tell you and what would make you abort?* — Circuit-only
    retention without Trotter layers: median 0.91 on ibm_boston, but four qubits below 0.4 and two
    with the wrong sign, and packet contrast 0.17 of ideal — an abort/relayout signal before spending
    the 32-s main run. The threshold used in `organizer/fallback_data/canary_*.json`: ≤ 2 flagged
    qubits, contrast retention ≥ 0.15.
20. ★★ *Your improvement run: what did you change and what is the z-score?* — Any defensible answer
    with z = (metric_new − metric_old)/sqrt(σ_new² + σ_old²) from the propagated uncertainties;
    z > 2 is a real improvement, |z| < 1 is "we quantified that it does not help", which is also
    accepted for full marks if the error bars are right. Options that actually help in the cached
    data: TREX readout mitigation + ODR (RMSE_W 0.052 vs 0.125); ZNE gate folding alone (0.104) but
    ZNE + ODR double-corrects (0.313).

---

## 7. Hint policy

* **Level 1**: inline in the participant notebook (`# hint:` lines and the "What to check" boxes).
  Mentors may repeat them freely.
* **Level 2**: released Day 1 at 18:00 (Slack/board) for the two exercises that block the hardware
  run. Mentors may not give more than these before release.
* **Level 3** (Day 2 after 12:00, mentors only, −1 point on the exercise): show the reference call
  signature and the shape of the return values, never the body.

**Level-2 hint, ex 2.2 (the junction fix).** "Count CZ gates in `circuits_all_isa[0]` (physics) and
`circuits_all_isa[1]` (mitigation). If the mitigation circuit has noticeably fewer, draw the middle
of the *logical* mitigation circuit: the last forward kinetic layer `RXXplus(+dt/4)` is immediately
followed by the first backward layer `RXXplus(−dt/4)`, and the optimizer cancels them (and on O3 also
the neighbouring single-qubit gates), so the calibration circuit no longer sees the same noise. Stop
the cancellation at the turning point without changing any unitary — a `barrier()` between the two
halves does it — then re-transpile and check that `n_cz_mitig` is within a few percent of
`n_cz_physics`. `evolve_circuits_matched` must return the same physics circuit and a protected
mitigation circuit whose statevector is still the initial state (fidelity 1 − 1e-9)."

**Level-2 hint, ex 3.1 (ODR).** "Work per site in ⟨Z⟩, not in chi: chi = (−1)^j Z + 1, so
1 − chi = −(−1)^j Z and the retention factor is simply f_j = ⟨Z_j⟩_cal / ⟨Z_j⟩_exact, where 'exact' is
the statevector/MPS value for the *calibration* circuit (= the t = 0 state). Steps: (1) mirror-average
chi and f (site j with 2L − 1 − j); (2) drop sites with f ≤ threshold (0.01) — if both mirror partners
fail, output NaN; (3) chi_mit = 1 − (1 − chi)/f. For `odr_uncertainty`, draw N = 2000 samples
evs ~ Normal(evs, stds) for all four PUBs, push each through *your* `odr_mitigate`, report the std of
X. For `odr_bias`, expand a/b to second order: E[a/b] − ā/b̄ ≈ (ā/b̄)·σ_b²/b̄² (independent a, b); the
grader compares your formula with the Monte-Carlo mean on synthetic data to 20 %."

---

## 8. Pitfalls (from SPEC.md §D and the build)

1. **`vacuum_prep_rotate_OV_3` range bug.** The interior R_−(θ) layer runs over `k in range(1, L-1)`
   (interior even pairs only). `range(2, L-1)` (a bug in the organizer's earlier benchmark code)
   deviates from the QDC reference by 0.0065 at sites 1, 4, 63, 66 and *still passes* a loose
   t = 0 comparison; the grader fingerprints at L = 6, 8 catch it.
2. **Fake backend names.** `FakeKingston().name == "fake_kingston"`, not `"ibm_kingston"`. Grader and
   `select_chain` must accept both; `get_qubit_coordinates` is patched to infer the heavy-hex width
   from the qubit count (156 → 16, 133 → 15).
3. **Readout / gate error `None`.** On real backends `backend.target["measure"][(q,)].error` (and
   occasionally CZ errors) can be `None` for a qubit; treat as 0 for ESP or exclude the qubit —
   never sort a list containing `None`. The three fake snapshots currently have no `None` entries, so
   code that works offline can still crash live.
4. **Local testing mode ignores every option except `shots`.** Running `EstimatorV2(mode=FakeKingston())`
   with twirling/DD/resilience options set does *not* twirl or add DD; the noisy rehearsal (ex 3.3)
   uses Aer with a hand-built Pauli `NoiseModel` (already Pauli, so it needs no twirling) and ex 3.2 is graded on
   the options object itself. Do not let a team "verify" twirling on a fake backend.
5. **O1 vs O3 layer structure.** Optimization level 3 re-synthesises the 2-qubit blocks, changes the
   CZ count (4,958 on FakeKingston at O3, seed 42, vs 5,292 at O1 — the structure the cached Kingston data used) and can
   alter the junction cancellation; use the *same* level for all four circuits and record it in
   `flight_plan`. Layout must be passed explicitly (`initial_layout=` + `layout_method="trivial"`) when
   re-transpiling onto `select_chain`'s path; O3 with `seed_transpiler` fixed otherwise finds its own
   chain.
6. **Do not use sessions.** Not available on the Open Plan; job or batch mode only. `Batch` is fine
   for the canary + main pair if a team wants them back-to-back.
7. **Never split the 4 PUBs across accounts.** One job, one layout, one calibration state; the grader
   requires one job id per run and identical `initial_layout` on all four ISA circuits.
8. **MPS bond-20 artefact.** The QDC repo's bond-20 t = 8 file is not converged (differs from a fresh
   bond-20 run by 1.5); only the bond-40 files are shipped here. Bond 40 is fine for participant runs
   (t = 8 in ≈ 30 s for two circuits); scoring uses the organizer bond-64 arrays.
9. **FakeKingston dead CZ edges.** The snapshot has CZ error 1.0 on (83,96), (96,103), (112,113),
   (120,121), (130,131), (145,146), (146,147). The *real* July-2026 Kingston layout in
   `hardware_ibm_kingston_2026-07-25.npz` runs through 103→96→83, so re-transpiling the cached layout
   onto `FakeKingston` fails or routes with swaps; use the frozen calibration summary in
   `grader_refs.npz` for 2.3, not a live `FakeKingston()` (FakeFez has 7 dead edges, FakeMarrakesh 13).
10. **Noise-model timing.** `NoiseModel.from_backend(FakeKingston())` at L = 6 takes ≈ 130 s with
    relaxation and 640 s in density-matrix mode; the reduced hand-built Pauli model
    (`reduced_noise_model`, depolarizing per CZ/sx from `backend.target` + readout) runs a 4-PUB
    L = 6 job in ≈ 30 s. Keep 3.3 on the reduced model.
11. **`evs` sign conventions.** The cached npz stores raw ⟨Z_j⟩, *not* chi; chi = (−1)^j Z + 1.
    Teams that measure `observables_isa` (chi directly) get chi. Mixing the two flips the profile.
12. **Angles ≈ 0 in the vacuum arm.** The vacuum circuit applies the wavepacket layers with
    θ = 0.9e-4 so the transpiler does not remove them; the cached Kingston data used 1e-4. Both are
    below every tolerance.
13. **Grader call cells must stay identical** in both notebooks; `build_notebooks.py` copies them
    verbatim, so an edit in one place propagates — never edit a built `.ipynb` by hand.

---

## 9. Regenerating everything

All commands from the project root in the environment built from `requirements.txt` (Python
3.11–3.13, Qiskit 2.5.2, runtime 0.49.0, Aer 0.17.2). Timings are on an idle 8-core MacBook.

| command | output | time |
|---|---|---|
| `python tools/test_reference.py` | verifies `schwinger_reference.py` (must print `ALL TESTS PASSED`) | ≈ 2–4 min (t = 8 bond-40 pair ≈ 30 s) |
| `python tools/make_mps_reference.py` | `reference_data/mps_reference_L34.npz` + `_meta.json` (bond 64 and 128, t = 2, 4, 6, 8) | ≈ 30–50 min (bond 64 t = 8: 110 s; bond 128 t = 8: ≈ 15 min) |
| `python organizer/make_grader_refs.py` | `reference_data/grader_refs.npz` (public references + the sealed answer key; copies `X_ref_t8_bd64`, `C_ref`, `RMSE_0` from `mps_reference_L34.npz`, so run it *after* `make_mps_reference`) | ≈ 30 s |
| `python organizer/make_bonus_b1_reference.py` | `reference_data/mps_reference_L34_dt.npz` (bond-32 profiles for dt = 1, 0.5, 0.25: the B1 answer, stripped from the participant kit) | minutes (three MPS pairs at bond 32) |
| `python organizer/make_hardware_fallback.py` | `organizer/fallback_data/hardware_fallback.npz` + `canary_fallback.npz` from `reference_data/hardware_ibm_kingston_2026-07-25.npz` and the canary json | seconds |
| `python tools/build_notebooks.py` | both notebooks, unexecuted | seconds |
| `python tools/build_notebooks.py --execute` | executes the solution notebook in place (`nbconvert`, kernel `python3`, per-cell timeout 3600 s, `MPLBACKEND=Agg`) and fills `submission/` | ≈ 3–7 min (SPEC target < 25 min) |
| `python tools/test_grader.py` | grader self-test: a perfect participant scores full marks, wrong variants lose points, regression cases (must print `ALL GRADER TESTS PASSED`) | ≈ 1 min (3–4 min with `--no-cache`) |
| `MPLBACKEND=Agg python tools/smoke_part_a.py`, `… smoke_part_b.py` | executes every cell of each notebook half in the solution view with a stub grader and checks the participant view for leaked answers (must print `SMOKE TEST PASSED`) | ≈ 1.5 min / ≈ 4 min |
| `python tools/make_kits.py` | `../schwinger-hadron-challenge-participant.zip` and `-organizer.zip`; asserts that the participant zip contains no solution or organizer material | seconds |
| `MPLBACKEND=Agg python tools/trotter_ordering_check.py` | the two ordering deviations of §12 (pinned odd-first step vs the even-first alternative at bond 40) | ≈ 1 min |

Order for a fresh machine: `test_reference` → `make_mps_reference` → `make_grader_refs` →
`make_bonus_b1_reference` → `build_notebooks --execute` → `test_grader` → `smoke_part_a` /
`smoke_part_b` → `make_kits`. `mps_reference_L34.npz` and `grader_refs.npz` must be regenerated after
any change to `trotter_step`.

The participant kit (`tools/make_kits.py`) is the project folder minus `organizer/`, `tools/`,
`scratch/`, `submission/`, the solution notebook and four organizer-only reference files:
`hardware_ibm_kingston_2026-07-25.npz` + `_provenance.json` (released with the fallback dataset),
`mps_reference_L34_dt.npz` (the B1 answer) and `make_mps_reference.log`.

---

## 10. Provenance and licensing

* Base material: `qiskit-community/qdc-challenges-2025`, Track B "Hadron dynamics in the Schwinger
  model" (Apache-2.0; copy of the licence in `LICENSE-qdc-challenges-2025`). `challenge_utils.py`
  (barbell, electric layer, `RXXplus`, `postselection_and_mitigation`, heavy-hex plotting), the seven
  figures in `images/` and the four L = 34 MPS text files are taken from it with attribution; edits
  are marked `# FF:`. Physics from Farrell, Illa, Ciavarella, Savage, arXiv:2401.08044 and
  arXiv:2308.04481 (SC-ADAPT-VQE angles 0.30738, −0.04059, −1.6492, −0.3281).
* `reference_data/hardware_ibm_kingston_2026-07-25.npz` + `_provenance.json`: raw ⟨Z_j⟩ and stds
  from the organizer's own July-2026 benchmark campaign (`benchmarkSchwinger` v6r2) on
  **ibm_kingston**, 2026-07-25, Qiskit 2.5.0 / runtime 0.47.0, O1 transpilation, layout starting
  143, 136, 123, …; strategies baseline, dd_only, twirl_only, twirl_dd, odr, trex, zne_fold, pec at
  T = 1, 2, 4, 6, 8; 100 twirls × 1000 shots. Job ids are in the provenance file (T = 8 ODR set:
  `d9hr3p50k0jc738il8dg`). `organizer/fallback_data/canary_ibm_boston_2026-07-27.json`: t = 0 canary
  on ibm_boston, 20,000 shots, job `d9jm0doii2cc73ef1p60`.
* If the organizers prefer data recorded under the event's own account, re-record the four ODR
  circuits at T = 8 with 64 twirls × 256 shots: ≈ 32 s of QPU, or ≈ 3 min for the full
  100 × 1000 configuration; `organizer/make_hardware_fallback.py --from-job <id>` repackages a job.
* The new material (SPEC, reference module, grader, notebooks, this guide) is released under the
  same Apache-2.0 terms; participant submissions remain the teams' property (QDC rule 5.1 carried over).

---

## 11. Answer key at a glance

| quantity | value | where |
|---|---|---|
| Model | m = 0.5, g = 0.3, L = 34 (68 qubits), t = 8, 8 Trotter steps of dt = 1 | fixed |
| E0 (full H, with +m/2·I) L = 4 / 6 / 8 | −1.18587 / −1.84744 / −2.50901 | ex 1.1 |
| E_ADAPT L = 4 / 6 / 8 | −1.17953 / −1.83805 / −2.49657 (gaps 0.0063 / 0.0094 / 0.0124) | ex 1.3 |
| Vacuum fidelity L = 4 / 6 / 8 | 0.9961 / 0.9945 / 0.9929 | ex 1.3 |
| Electric layer identity | exact to 1e-8 (L = 4 and 6, both barbell branches) | ex 1.2 |
| One-step Trotter error L = 6, dt = 0.2 / 0.1 / 0.05 | 1.85e-3 / 2.37e-4 / 2.99e-5, ratios 7.8, 8.0 | ex 1.4 |
| max chi Trotter error L = 6, t = 8, dt = 1 / 0.5 | 0.133 / 0.061 | ex 1.4 (indicative) |
| Logical physics circuit | 5,758 2q gates, 2q-depth 256; FakeKingston O3 (seed 42): 4,958 CZ, 2q-depth 220, swap-free (O1: 5,292 CZ) | ex 2.1–2.2 |
| MPS bond 40 vs QDC bond-40 file (pinned odd-first ordering, t = 8) | max|ΔX| < 5e-5 (even-first: 0.023, nrmse 0.048, Pearson 0.9987) | ex 1.5 / §12 |
| MPS bond 64 (pinned) vs QDC bond-40 file, t = 8 | max|Δchi| 0.019, max|ΔX| 0.0008 | ex 1.5 |
| MPS convergence t = 8: max|Δ| bond 64 vs 128 (even-first run) | chi 0.015, X 0.0007 | ex 1.5 |
| MPS timings (t = 8, two circuits, 8 cores) | bond 40 ≈ 30 s, bond 64 = 110 s (pinned; 369 s in the even-first run), bond 128 = 887 s (even-first run) | ex 1.5 |
| X_ref (bond 64, pinned ordering) sites 31 / 32 / 33 / 34 / 35 / 36 | 0.509 / 0.507 / −0.006 / −0.006 / 0.507 / 0.509 (stale even-first file: 0.525 / 0.496 / −0.002) | ex 4.2 |
| RMSE_0 (all-zero submission), C_ref | 0.269, 0.514 (SPEC constant 0.515; stale even-first file: 0.270, 0.512) | ex 4.2 |
| Cached Kingston raw (twirl + DD) | RMSE_W 0.229, C 0.068, quiet 0.003 | ex 4.2 calibration |
| Cached Kingston ODR | RMSE_W 0.125, C 0.437, quiet 0.021, retention 0.14–0.33 (median 0.21), centre 0.247 / −0.132 / −0.132 / 0.247 | ≈ 12/18 |
| Cached Kingston other strategies + ODR (RMSE_W) | trex 0.052, twirl_only 0.116, twirl_dd 0.134, baseline 0.167, dd_only 0.235, zne_fold 0.313 | ex 4.3 ideas |
| Cached ODR at T = 2 / 4 / 6 (RMSE_W vs bond 64) | 0.099 / 0.111 / 0.088 | |
| Canary (ibm_boston, t = 0) | median retention 0.91; 4 qubits < 0.4; packet contrast 0.17 of ideal → "fail" | ex 4.1 |
| Usage | canary ≈ 4 s, main ≈ 32 s, cap 180 s; QDC config ≈ 348 s | ex 2.4 / 4.x |
| Trotter ordering at bond 40 vs QDC file | pinned (Fig. 8, odd-first): < 5e-5; even-first: 0.023 (nrmse 0.048) | §12 |

---

## 12. Known deviations from SPEC.md (found while writing this guide)

1. **Trotter ordering — measured deviations (the "builder task" of SPEC §E).** With Aer MPS at bond
   40 (seed 7), L = 34, t = 8, 8 steps of dt = 1 (`tools/trotter_ordering_check.py`):
   odd-first opening kinetic half-step (the pinned Fig. 8 ordering after the 2026-09-07 correction):
   max|X − X_QDC40| < 5e-5 (bit-identical to the QDC file); even-first: max|X − X_QDC40| = 0.0234,
   nrmse 0.048, Pearson 0.9987, centre 0.525 / 0.496 / −0.002 vs 0.510 / 0.508 / −0.006. The first
   draft of SPEC §E and of `schwinger_reference.trotter_step` had even-first; both were corrected
   before the grader build. **Resolved 2026-09-10:** `reference_data/mps_reference_L34.npz` and the
   `X_ref_t8_bd64` / `C_ref` / `RMSE_0` copies inside `grader_refs.npz` were regenerated with the
   pinned odd-first step and now match the QDC bond-40 file to 8e-4 (`tools/test_reference.py` step
   [7] reports nrmse 3.5e-5).
2. `X_ref` constants: the shipped bond-64 reference has peaks 0.509 / 0.507, dip −0.006,
   C_ref = 0.514, RMSE_0 = 0.269 — consistent with SPEC §E (0.51, −0.006, 0.515, 0.269).
   Since 2026-09-10 `fallfest_grader.hardware_metric` *measures* C_ref from the same profile it
   scores against (0.514) instead of hard-coding SPEC's 0.515, and the notebook's
   `leaderboard_preview` uses the same number, so preview and grader agree exactly.
3. Calibration of ex 4.2: SPEC assumes the cached ODR result loses the quiet-region 2 points
   ("quiet 0.15 → 0"); with quiet defined as the median of |X| outside W it is 0.021 ≤ 0.05 and the
   cached result scores ≈ 12/18, not ≈ 10/18. If 10/18 is the intended calibration, define quiet as
   the *maximum* (0.15) instead of the median. [Integration decision 2026-09-07: the median (the SPEC
   formula as written) is kept; the SPEC calibration line now reads ≈ 12/18.]
4. SPEC §B now says the solution notebook must execute in < 25 min (an earlier draft said < 20). The
   executed solution takes 3–7 min because it loads the organizer bond-64 arrays instead of
   recomputing them (110 s at t = 8 on 8 cores).
5. `tools/build_notebooks.py` reads the cell list from `tools/nb_part_a.py` + `tools/nb_part_b.py`
   (not from itself as SPEC §B says); both files are part of the organizer kit.
6. `hardware_ibm_kingston_2026-07-25.npz` ships *all* strategies (including trex / zne_fold arrays that
   score better than the plain ODR set); SPEC only mentions the ODR circuits. Hence integrity check
   4 in §5.
7. The cached Kingston layout crosses CZ edges that are dead in the `FakeKingston` snapshot (pitfall 9);
   ex 2.1/2.3 grading must use the frozen target summary, not the cached layout on the fake backend.
