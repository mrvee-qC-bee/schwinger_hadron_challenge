# Build specification — "Hadron Dynamics in the Schwinger Model", UofT Qiskit Fall Fest edition

Status: fixed sections (A–D) are final; section E (exercise list) is filled in from the design
panel and is the contract for the notebook builder, the grader and the organizer guide.

## A. Deliverables and file layout (`schwinger-hadron-challenge/`)

Participant kit (ship this folder minus `organizer/` and `tools/`):

| path | purpose |
|---|---|
| `schwinger_hadron_participant.ipynb` | the challenge notebook with `# PROMPT` / `# BEGIN ANSWER` blanks |
| `challenge_utils.py` | given helpers: `barbell`, `trotter_step_electric_2q`, `RXXplus`, `postselection_and_mitigation`, heavy-hex plotting (`get_qubit_coordinates`, `plot_qubit_chain`) — adapted from QDC (Apache-2.0, attributed) |
| `fallfest_grader.py` | LOCAL autograder (no network). Each `grade_exN(...)` prints PASS/FAIL + points and appends to `submission/score.json` |
| `reference_data/` | QDC MPS text files (t=0 exact, t=8 bond-40), `mps_reference_L34.npz` + `_meta.json` (bond 64/128, t=2,4,6,8), `grader_refs.npz` (public references + the sealed answer key) |
| `images/` | QDC figures (Apache-2.0) |
| `requirements.txt` | qiskit~=2.5, qiskit-ibm-runtime~=0.49, qiskit-aer~=0.17, numpy, scipy, matplotlib, seaborn, jupyter |
| `README.md` | participant-facing: setup, rules, submission format, QPU budget rules |

Organizer kit (adds):

| path | purpose |
|---|---|
| `schwinger_hadron_solution.ipynb` | fully solved AND executed (outputs included); hardware step uses cached results unless `RUN_ON_HARDWARE = True` |
| `reference_data/hardware_ibm_kingston_2026-07-25.npz` + `_provenance.json`, `mps_reference_L34_dt.npz` | organizer-only data: cached real-QPU raw ⟨Z⟩ (4 ODR circuits at T=2,4,6,8 and other strategies; released with the Day-2 fallback dataset) and the Bonus-B1 answer profiles |
| `organizer/schwinger_reference.py` | verified reference implementation (source of truth; imported by the grader-reference generator and the builder's solution cells) |
| `organizer/make_grader_refs.py` | regenerates `reference_data/grader_refs.npz` |
| `organizer/grade_submission.py` | CLI: re-scores a team zip / `submission/` folder (arrays, hardware metric, `team_functions.py` re-runs, tamper check against the executed notebook) → JSON + leaderboard line / CSV |
| `organizer/ORGANIZER_GUIDE.md` | rubric, timeline, QPU plan, judge sheet for open-ended parts, pitfalls, answer key summary |
| `organizer/SPEC.md` | this file |
| `tools/build_notebooks.py` | single source of truth → participant + solution notebooks |
| `tools/test_reference.py`, `tools/make_mps_reference.py` | verification + reference generation |

## B. Single-source notebook build

`tools/build_notebooks.py` holds the notebook as an ordered list of cells built with helpers
`md(text)` and `code(src, *, tags=())`.

Conventions inside code cells:

* `# PROMPT: ...` — instruction line, kept in both versions.
* `# BEGIN ANSWER` … `# END ANSWER` — solution lines. In the participant build the block body is
  replaced by `    # YOUR CODE HERE` (indentation preserved from the `# BEGIN ANSWER` line) unless a
  line inside is marked `# KEEP` (scaffold that participants should see).
* `name = # <type/description>` lines (QDC style) are allowed in the participant version; the
  solution version has the full assignment.
* Cell tags: `solution-only` (dropped from participant build; e.g. organizer sanity checks),
  `participant-only` (dropped from solution build), `hardware` (cell that would submit a job; it must
  be guarded by `if RUN_ON_HARDWARE:` and fall back to cached data).
* Every grader call cell `grade_exN(...)` is identical in both versions.

Build + execute (organizer machine, conda env `qiskit-paper`):

```
python tools/build_notebooks.py                # writes both notebooks (unexecuted)
python tools/build_notebooks.py --execute      # additionally executes the solution notebook in place
                                               # (jupyter nbconvert --execute, kernel python3, timeout 3600)
```

The executed solution must run start-to-finish offline (no IBM credentials) in < 25 min on a laptop.
`RUN_ON_HARDWARE = False` is the default in both notebooks.

## C. Autograder contract (`fallfest_grader.py`)

* Pure-Python + numpy + qiskit; loads `reference_data/grader_refs.npz` lazily; never imports the
  organizer module.
* Every `grade_exN` returns the points awarded (float) and prints a one-line verdict with a hint on
  failure. Points are written to `submission/score.json` as `{"exN": {"points": p, "max": m, "detail": str}}`.
* Anti-hard-coding: circuit-building exercises are graded by CALLING the participant's function at a
  different, smaller L (and, where relevant, at a random angle) and comparing against stored
  reference statevector fingerprints/expectation arrays; numeric-array exercises are compared with
  tolerances; hardware-quality exercises are scored from arrays saved in `submission/`.
* Deterministic: fixed seeds, no dependence on transpiler randomness beyond what is checked
  structurally (ISA compliance vs `backend.target`, layout equality, 2q-depth ranges).
* `organizer/grade_submission.py` re-runs the same checks on a submitted folder (or `<team>.zip`) and adds the
  hardware leaderboard metric. Array-graded exercises are re-scored from `submission/exN.npz`; function-level
  exercises are re-run from `submission/team_functions.py`, which the notebook's last code cells export
  (every notebook function plus its constants, test-imported on the spot) -- without it their local points
  are kept provisionally and listed as not re-verified for the viva. `score.json` is cross-checked against
  the `[exN] p/m` lines in the executed notebook (tamper check). `--hidden` re-runs `select_chain` on FakeFez
  and FakeMarrakesh.

## D. Verified physics facts (from `tools/test_reference.py`, all PASS)

* `RXYplus(θ) = exp(-iθ/2 (XY+YX))`, `RXYminus(θ) = exp(+iθ/2 (XY−YX)) = exp(iθ O)` with `O=(XY−YX)/2`,
  `RXXplus(θ) = exp(-iθ/2 (XX+YY))`.
* Electric layer (single-qubit Rz layer + `trotter_step_electric_2q`) equals `exp(-i t H_el^{(Q=0)}(1))`
  exactly (both barbell tiling branches, L=4 and L=6) with `H_el^{(Q=0)}(1)` transcribed from the
  notebook equation (`schwinger_reference.electric_hamiltonian_truncated`).
* `wave_prep_rotate_O_11/O_22` implement `exp(+iθ O_mh(1,1))`, `exp(+iθ O_mh(2,2))` exactly; the two
  3-site pieces of `O_mh(2,2)` commute.
* The 2-step SC-ADAPT-VQE vacuum (angles 0.30738, −0.04059) has exact-ground-state fidelity
  0.9961 (L=4), 0.9945 (L=6), 0.9929 (L=8) w.r.t. the FULL (untruncated) Hamiltonian, energies
  E0 = −1.18587 / −1.84744 / −2.50901 vs E_ADAPT = −1.17953 / −1.83805 / −2.49657 (with the `+m/2·I` per site included).
* One second-order Trotter step vs `exp(-i dt H_trunc)` at L=6: error 1.85e-3 / 2.37e-4 / 2.99e-5
  for dt = 0.2 / 0.1 / 0.05 (ratios 7.8, 8.0 → third-order local error confirmed).
* `vacuum_prep_rotate_OV_3` must apply the interior `R_-(θ)` layer for `k in range(1, L-1)`; the
  benchmark project's `range(2, L-1)` deviates from the QDC reference by 0.0065 at sites 1, 4, 63, 66.
* L=34: t=0 exact chi matches QDC files to 1e-11; t=8 bond-40 MPS matches the QDC bond-40 files to
  < 5e-5 in the vacuum-subtracted profile with the Fig.-8 (odd-first) kinetic ordering (the even-first
  variant deviates by 0.023, nrmse 0.048, Pearson 0.9987). Physics circuit: 5758 2q
  gates, 2q-depth 256 logical; on FakeKingston O3 (seed 42): 4958 CZ, 2q-depth 220, swap-free 68-qubit chain
  (O1: 5292 CZ). [Numbers corrected 2026-09-07 for the odd-first ordering; the earlier 5756/5294/242 were even-first.]
* Aer MPS timings (MacBook, 8 cores): t=0 in 3 s; t=8 bond 40 ≈ 12–40 s per circuit; bond 64 t=2 ≈ 55 s for two circuits.
* Cached ibm_kingston (2026-07-25, job d9hr3p50k0jc738il8dg for T=8 ODR set, 100 twirls × 1000 shots,
  DD XY4, resilience_level 0): raw vacuum-subtracted profile at t=8 → nrmse 0.85 / Pearson 0.975 vs
  QDC reference (amplitude suppressed, shape right); after ODR → nrmse 0.55 / Pearson 0.86 with the
  central dip and side peaks visible (centre values 0.247, −0.132, −0.132, 0.247 vs MPS 0.507, −0.006, −0.006, 0.507).
  These numbers calibrate the hardware leaderboard thresholds.
* Fake backends usable offline: FakeKingston / FakeFez / FakeMarrakesh (the three Open-Plan Heron r2
  devices), `get_qubit_coordinates` patched to accept them.
* Open Plan facts (docs, 2026-09-07): 10 min QPU per 28-day rolling window per instance; +180 min
  opt-in promo for active users (since 16 Mar 2026); job and batch modes only (no sessions);
  Classroom Accounts give each student an Open-Plan instance without a credit card.


## E. Exercise contract (final)

Design decisions (from the three-lens panel + two judges, resolved by the author):

* Points come from content that is NOT in the public QDC notebook; the QDC fill-ins are a 10-point
  warm-up graded by calling the participant's *functions* at other lattice sizes.
* No displaced-packet variant: the packet stays centred (CP mirror averaging is kept as a tool).
  Anti-copy = function-level grading at random L + hand-written ODR + the Runtime-options exercise + hardware + viva.
* Main hardware run uses runtime `EstimatorV2` twirling (evs + stds); uncertainties by Monte-Carlo
  propagation of the Estimator stds through the team's own ODR. Twirling, DD and any readout/ZNE/PEC mitigation go through
  `EstimatorOptions` built by the team's `mitigation_options` (ex 3.2); hand-rolled twirling or bitstring post-selection is
  not accepted, in the improvement run either.
* Hamiltonian convention: `H = (m/2) Σ_j [(-1)^j Z_j + I] + (1/2) Σ_j (σ+_j σ-_{j+1} + h.c.) + (g²/2) Σ_j (Σ_{k≤j} Q_k)²`
  WITH the `+m/2·I` per site, so `E0(L=8) = -2.50901` (the panel's −6.50901 dropped the 4.0 constant).
  The grader compares Pauli coefficients with the identity term excluded, so either convention passes
  the Hamiltonian check; reported energies must follow the stated convention.
* Trotter ordering is pinned to Fig. 8: `Hkin(t/2)[odd bonds (1,2),(3,4),.. , even bonds (0,1),(2,3),..] · Hel(t)[Rz layer, barbells] · Hm(t) · Hkin(t/2)[even, odd]`
  (this is what `schwinger_reference.trotter_step` does and what the MPS reference uses).
  [Corrected 2026-09-07 by the grader build: an earlier draft of this line and of `trotter_step` had
  even-first; Fig. 8 (zoomed), the QDC bond-40 file (match < 5e-5 vs 0.023) and the cached Kingston
  run all use odd-first.] The grader
  runs a second-order ratio test (any valid ordering passes) AND a fidelity check against the pinned
  ordering (required, because the hardware reference is computed for that ordering). Builder task
  (DONE): confirmed by MPS (bond 40) that the pinned odd-first ordering matches the QDC bond-40 t=8
  file (< 5e-5) better than the alternative even-first ordering (0.023); both deviations are recorded
  in the organizer guide.
* Noisy rehearsal uses a reduced, hand-built Pauli noise model from `backend.target` (fast, ~30 s per
  4-PUB run at L=6), not `NoiseModel.from_backend` (130 s with relaxation, 640 s density-matrix).
* Reference for hardware scoring: organizer bond-64 MPS (`reference_data/mps_reference_L34.npz`,
  keys `chi_wave_t8_bd64`, `chi_vacuum_t8_bd64`; bond-128 copies for validation). Window
  `W = {25,…,42}`; `X_ref[W]` peaks 0.51 (sites 31,32,35,36), dip −0.006 (33,34), `RMSE_0 = 0.269`.
* Hardware metric calibration (cached ibm_kingston 2026-07-25, 5292 CZ at O1, 100 twirls × 1000 shots):
  raw (twirl+DD) RMSE_W 0.23, contrast 0.06; ODR RMSE_W 0.126, contrast 0.437, retention 0.15–0.33.

### Points (100 + bonus)

| Part | id | name | pts | graded by |
|---|---|---|---|---|
| 0 Warm-up (public fill-ins, function-level) | 0.1 | `chiral_condensate_observables(L)` | 2 | auto |
| | 0.2 | `RXYplus(theta)`, `RXYminus(theta)` | 2 | auto |
| | 0.3 | `prep_strong_coupling_vacuum(L)`, `vacuum_prep_rotate_OV_3(qc, theta, L)`, `prep_vacuum(L, th1, th3)` | 2 | auto |
| | 0.4 | `wave_prep_rotate_O_22(qc, theta, L)`, `prep_wave(L, th1, th3, th11, th22)` | 2 | auto |
| | 0.5 | `trotter_step(qc, L, time_step, m, g)`, `evolve_circuits(qc_init, L, t, m, g, n_steps=None)` | 2 | auto |
| 1 Physics you can verify | 1.1 | `schwinger_hamiltonian(L, m, g)`, `electric_hamiltonian_truncated(L, g)`, charge sector checks, `E0_L8` | 5 | auto |
| | 1.2 | `electric_layer(L, t, g)` == `exp(-i t H_el^(1))` (infidelity at L=4,6) + `truncation_shift` (L=8, t=4, exact evolution full vs truncated H) | 4 | auto |
| | 1.3 | `vqe_fidelity`, `vqe_energy_gap` at L=4,6,8 vs exact diagonalization | 3 | auto |
| | 1.4 | Trotter error table at L=8, t∈{2,4}, dt∈{1,0.5,0.25} vs exact evolution of the truncated H; ratios; Richardson; `dominant_error` | 5 | auto |
| | 1.5 | MPS bond-dimension convergence at L=34, t=8 (`mps_err` dict, wall times), why the QDC bond-20 file is unusable | 3 | auto |
| 2 Engineering the 68-qubit experiment | 2.1 | Transpile 4 circuits + observables to one layout (`circuits_all_isa`, `observables_isa`) | 3 | auto |
| | 2.2 | Gate accounting (`n2q_logical`, `n_cz_physics`) + noise-matched calibration circuit `evolve_circuits_matched(...)` | 5 | auto |
| | 2.3 | `select_chain(backend, n_qubits=68)` calibration-aware path search + re-transpile onto it | 5 | auto |
| | 2.4 | `flight_plan` dict + `make_estimator_options(backend)` + usage prediction | 2 | auto |
| 3 Mitigation you wrote yourself | 3.1 | `odr_mitigate(...)`, `odr_uncertainty(...)`, `odr_bias(...)` on synthetic data | 7 | auto |
| | 3.2 | `mitigation_options(base_options, num_randomizations, shots_per_randomization, dd_sequence, readout_mitigation)` → Runtime `EstimatorOptions` (twirling, DD, resilience) | 5 | auto |
| | 3.3 | Noisy rehearsal at L=6, t=4: `reduced_noise_model(backend, chain)`, run, own ODR, RMSE vs exact | 8 | auto |
| 4 Hardware | 4.1 | Canary at t=0 (2 PUBs × 16 twirls × 128 shots, ≤ 30 s usage) | 3 | auto |
| | 4.2 | Main t=8 run (4 PUBs × 64 twirls × 256 shots) → leaderboard metric | 18 | auto |
| | 4.3 | Improvement run (≤ 60 s usage) with z-score | 4 | auto |
| 5 Error budget + defence | 5 | one-page error-budget table, ODR bias derivation, 8-min pitch, viva | 10 | judges |
| Bonus (max +6, not in 100) | B1 | L=34 dt→0 Richardson (dt=0.5, 0.25 at bond 32) | 3 | auto (structural) + organizer arrays |
| | B2 | ODR on amplitude-damping vs depolarizing toy at L=4 with/without twirls; witness ordering | 2 | auto |
| | B3 | Fractional-gate barbell (`rzz`) unitary check | 1 | auto |

### Grader API (`fallfest_grader.py`)

```python
import fallfest_grader as ff
ff.check_env()                                   # prints versions, warns if not qiskit>=2.5 / runtime>=0.49
ff.grade_ex0_1(chiral_condensate_observables)    # every grader takes FUNCTIONS or arrays, returns points (float)
ff.grade_ex0_2(RXYplus, RXYminus)
ff.grade_ex0_3(prep_vacuum)                      # calls prep_vacuum(L, 0.30738, -0.04059) at L in {6, 8}
ff.grade_ex0_4(prep_wave)                        # prep_wave(L, th1, th3, th11, th22)
ff.grade_ex0_5(trotter_step, evolve_circuits, prep_wave)
ff.grade_ex1_1(schwinger_hamiltonian, electric_hamiltonian_truncated, E0_L8)
ff.grade_ex1_2(electric_layer, truncation_shift)          # electric_layer(L, t, g) -> QuantumCircuit
ff.grade_ex1_3(vqe_fidelity, vqe_energy_gap)              # dicts {4: .., 6: .., 8: ..}
ff.grade_ex1_4(trotter_table, richardson_error, dominant_error)   # trotter_table {(t, dt): max_abs_err}
ff.grade_ex1_5(mps_err, mps_seconds, X_bond40)            # mps_err {bond: max|X(bond) - X(64)|}
ff.grade_ex2_1(circuits_all_isa, observables_isa, backend)
ff.grade_ex2_2(n2q_logical, n_cz_physics, evolve_circuits_matched, prep_wave, backend, layout)
ff.grade_ex2_3(select_chain, backend)                     # select_chain(backend, n_qubits=68) -> list[int]
ff.grade_ex2_4(flight_plan, estimator_options, circuits_all_isa)
ff.grade_ex3_1(odr_mitigate, odr_uncertainty, odr_bias)
ff.grade_ex3_2(mitigation_options)
ff.grade_ex3_3(noise_model, chain, chi_raw_arrays, X_raw, X_mit, odr_mitigate, backend)
ff.grade_ex4_1(canary_result, job_info)                   # dict with evs arrays; job_info {job_id, backend, usage_s, ...}
ff.grade_ex4_2(hardware_result, job_info, odr_mitigate)   # hardware_result: evs/stds for the 4 PUBs, X_raw, X_mit, sigma
ff.grade_ex4_3(improvement, odr_mitigate)                 # dict with both runs + rationale
ff.grade_bonus_B1(...); ff.grade_bonus_B2(...); ff.grade_bonus_B3(...)
ff.summary()                                              # prints the score table from submission/score.json
```

Every grader: (1) validates types, (2) runs the check with fixed seeds, (3) prints `[ex1.2] 4.0/4 PASS — ...`
or `[ex1.2] 1.0/4 — hint: ...`, (4) stores points + a compact copy of the inputs in
`submission/score.json` / `submission/exN.npz` for organizer re-grading, (5) never raises on a wrong
answer (only on wrong types). Total runtime of all graders on a laptop < 3 min excluding 3.3/4.x
(which score saved arrays).

Grader oracles/reference data (`reference_data/grader_refs.npz`, produced by `organizer/make_grader_refs.py`):
reference statevectors for `prep_vacuum`/`prep_wave` at L=6,8 and the t=2 physics circuit at L=6;
exact ground states + energies at L=4,6,8; full and truncated Hamiltonian Pauli dictionaries at L=4,6,8;
truncation shift at L=8, t=4; Trotter table values; exact X at L=6, t=4 (for 3.3) and at L=8, t∈{2,4};
`X_ref` window constants; a frozen `FakeKingston` target summary (CZ errors per edge, readout errors,
sx errors) used by 2.3 and 3.3 so results do not drift with runtime releases. Physics oracles that are
cheap and generic (expm of a 2-qubit generator, `SparsePauliOp` algebra, Trotter ratio test) run on the
fly inside the grader.

### Hardware metric (ex 4.2, 18 pts)

Inputs: `X_hat` = grader's own recomputation from the submitted 4 PUB evs with the team's
`odr_mitigate` (must equal the submitted `X_mit` to 1e-6, else 0 points and a judge flag);
`X_ref = chi_wave_t8_bd64 - chi_vacuum_t8_bd64`; window `W = 25..42`.
`RMSE_W = sqrt(mean_{j∈W}(X_hat_j - X_ref_j)^2)` (an unreported / NaN window site is charged an error of max(|X_ref_j|, 0.25) for both points and rank, so hiding a site never helps; a NaN outside W counts as 1.0 in the quiet median; sites 31–36 must all be finite for the contrast and shape points; fewer than 16 finite window sites → NOT RANKED);
contrast `C = mean(X_hat[31,32,35,36]) - mean(X_hat[33,34])`, `C_ref = 0.515`;
`quiet = median_{j∉W} |X_hat_j|`; coverage `κ` = fraction of j∈W with `|X_hat_j - X_ref_j| ≤ 2 sqrt(σ_j² + 0.003²)`.

`points = 10·clip(1 − RMSE_W/0.25) + 4·clip(1 − |C − C_ref|/0.35) + 2·[C ≥ 0.20 and min(X_hat[31,32,35,36]) − max(X_hat[33,34]) ≥ 0.25] + 2·[quiet ≤ 0.05 or judge-accepted systematics note]`
Gates: 4 PUBs on one layout; usage ≤ 180 s (else −5); ODR factors < 0.9 on ≥ 10 window sites;
`fallback=True` entries (organizer-released dataset) capped at 75 % and flagged (the usage gates of 4.2/4.3 are
waived on fallback data, whose usage is only an organizer estimate; 4.3 is capped at 75 % as well).
Calibration: cached Kingston ODR → 5.0 + 3.1 + 2 + 2 (quiet = median |X| outside W = 0.021) ≈ 12/18
(an earlier draft quoted ≈ 10/18 by taking the maximum 0.15 for quiet); raw twirl+DD → ≈ 0–1/18.
Leaderboard: rank by RMSE_W, ties by full-68-site RMSE, then lower usage.

### QPU plan (per team, hard cap 180 s of usage)

Usage model: `2 s + 0.45 ms × executions` per job (documented 0.35 ms scaled to the observed 6-min QDC
run). Canary: 2 × 16 × 128 = 4,096 executions ≈ 4 s. Main: 4 × 64 × 256 = 65,536 ≈ 32 s.
Improvement: ≤ 60 s. Open Plan = 10 min per instance per 28 days on ibm_fez / ibm_marrakesh /
ibm_kingston, job or batch mode only. Fallback: organizer-recorded dataset (this repo ships the July-2026
ibm_kingston set) released on Day 2 morning to teams without a returned job, capped at 75 %.

### Notebook narrative (participant version, ~120 cells)

Title: "Hadron Dynamics in the Schwinger Model — Trust, but Verify (UofT Qiskit Fall Fest 2026 edition)".
Intro keeps the QDC physics text/figures (attributed), then a box "What is different from the public
QDC notebook and why", rules (QPU cap, submission format, viva), the roadmap table above, and
`ff.check_env()`. Parts 0–5 as listed; each exercise = short theory → `# PROMPT` code cell →
grader cell. Level-1 hints inline. `RUN_ON_HARDWARE = False` default; the hardware cells are guarded
and fall back to `reference_data/hardware_fallback.npz` when present (organizer-released), so the
solution notebook executes fully offline with the cached Kingston data.
