# Build notes — integration and review-fix pass (2026-09-10)

Every change below was made in the pinned environment (qiskit 2.5.2 / qiskit-ibm-runtime 0.49.0 /
qiskit-aer 0.17.2). No IBM Quantum jobs were submitted. The notebooks were never hand-edited: all
notebook changes are in `tools/nb_part_a.py` / `tools/nb_part_b.py` and the `.ipynb` files are rebuilt
by `tools/build_notebooks.py`. (`scratch/` files mentioned in these notes are working files and are not
part of the organizer kit.)

## 1. Review findings applied (`organizer/REVIEW_FINDINGS.json`)

### High

| # | Finding | Fix |
|---|---|---|
| 1 | `pm_pinned` used by given code in 2.2 but never requested | the 2.1 prompt now names `pm_pinned`, `circuits_all_isa` and `observables_isa` explicitly |
| 2 | `X_R` used by the 1.4 plot but never requested | the 1.4 prompt now names `X_trotter`, `trotter_table`, `X_R`, `richardson_error`, `dominant_error` |
| 3 | `grader_refs.npz` shipped the answers in clear text | the file is now split: 46 public keys (things participants legitimately need) + one sealed blob holding the 73 oracle keys. See §2. |
| 4 | `_pauli_dict` used `str(p)`, which truncates Pauli labels beyond 50 qubits | uses `p.to_label()`; a sign flip on a high-index observable is now caught (regression test added) |

### Medium

| # | Fix |
|---|---|
| 5 | one `RUN_ON_HARDWARE` switch (Part 0); the Part-4 cell prints its value instead of reassigning it |
| 6 | `load_fallback` returns `None` with a clear message instead of raising; both Part-4 cells then carry NaN arrays and a `no_data` job stub, so Part 5 and the bonus stay reachable when the organizer dataset has not been released |
| 7 | submission file names aligned across README, guide §5 and the notebook checklist; the notebook now writes `submission/job_info.json` (canary + main, extended by 4.3) as the README promised |
| 8 | one report format everywhere: `submission/report.md`, at most 2 pages including figures |
| 9 | the Part-5 judge sheet shown to participants is now the guide's 3 / 3 / 4 split |
| 10 | the 2.2 text states the graded criterion (CZ count within 1 %, per-edge within ±2) and says the barrier variant already passes |
| 11 | `job.usage()` returns 0 while the usage record is pending; a new `measured_usage()` helper polls `job.metrics()['usage']` (10 × 6 s), falls back to the planning model and sets `usage_estimated` |
| 12 | the B2 witness-ordering claim now matches the toy's own output (AD untwirled ≫ {AD twirled, depolarizing}) |
| 13 | canary circuits are ~540 CZ, not ~60 |
| 14 | the ODR-uncertainty check is no longer a seed lottery: the grader averages three independent 10⁴-sample Monte-Carlo streams and grades only well-conditioned sites (`f > 0.2`, denominator relative uncertainty < 0.2, three streams agreeing to 10 %) |
| 15 | `grade_ex4_3` no longer awards the improvement point on `z = inf`; with no uncertainties `z` is NaN, the message says so and only the documented-null half point is available |
| 16 | the leaderboard and the points are computed on the NaN-filled profile, so hiding bad sites can no longer help. See §3. |
| 17 | ex 0.5 additionally checks the mitigation circuit's 2q-gate count against the physics circuit; ex 2.2 grants the matched-circuit points only if the L = 6 return check also passes, and detects a mitigation circuit that is a copy of the physics circuit |
| 18 | plausibility judge flags added (job-id shape, RMSE below the shot-noise floor, coverage with sub-shot-noise sigma, Estimator stds inconsistent with the declared shots, perfectly uniform ODR factors). Flags never change the score. |
| 19 | `target_summary` raises a clear `ValueError` for a backend with no frozen snapshot and no live target; `grade_submission.py` catches it, keeps the local score and notes that the layout/chain was not re-checked |
| 20 | graders no longer raise on wrong-but-plausible input: new `_num` / `_safe_statevector` / `_evolve_safe` helpers, `_statevector` strips final measurements, `_norm_table` parses string keys with a regex |

### Low (applied)

21 (graded energy no longer printed next to its blank, and the "would be −6.50901" hint removed),
22 (`qc`, `H`, `best` and the `compare_runs` return contract are now named in the prompts/scaffolds),
23 + 29 (7 dead couplers, 14 directed entries; readout qubit 146), 24 (README gate count ≈ 5,000 at
optimization level 3), 25 + 32 (the preview and the grader use one measured `C_ref` and the same
NaN and fallback-cap rules), 26 (README time budget matches the guide's schedule), 27 (local testing
mode warns and does use the snapshot noise model; it simply cannot exercise mitigation options),
28 (`_qpy_load` raises `GraderRefError` with a "reinstall the pinned environment" hint; the refs
record `refs_qiskit_version` / `refs_qpy_version`), 30 (barbell has 12 CX), 31 (the amplitude-damping
offset statements in 3.1 and B2 reconciled), 33 (numpy scalars, size-1 arrays, numeric strings and
per-site `predicted_sigma_X` accepted), 34 (judge flags when the returned chain is the grader's own
baseline chain or the transpiler's default layout), 35 (flags when `X_bond40` is bit-identical to a
shipped file, or when the wall times are missing or do not grow with the bond dimension).

### Deferred / rejected

* **Finding 34, tightening the ex 2.3 cost tiers.** Tried 1.02 / 1.08 / 1.20 with a 4-seed baseline:
  it cost the *reference* solution a point on the hidden FakeFez backend (ratio 1.039) while the
  transpiler's default layout still passed on Kingston (ratio 1.011). Reverted to 1.05 / 1.15 / 1.30.
  The baseline is still strengthened (best of 4 seeds; Fez 22.519 → 21.986) and the gaming case is
  caught by the two judge flags plus the hidden-backend re-grade. A cost ratio cannot distinguish a
  search from a good default when the default *is* good.
* **Finding 3, hashing individual scalars instead of sealing.** A sha256 of a value rounded to the
  grading grid is brute-forceable over a plausible range, and hashing the Pauli dictionaries would
  destroy the coefficient-level diagnostics the reviewers singled out as the grader's best feature.
  The blob keeps every diagnostic and closes the realistic path (`np.load` + read).

## 2. The sealed oracle

`organizer/make_grader_refs.py` now splits its 119 keys:

* **public (46)** — `X_ref_t8_bd64/128`, `C_ref`, `RMSE_0`, the window/peak/dip constants, the frozen
  `tgt_*` target snapshots for Kingston/Fez/Marrakesh, the grader-owned ex-3.2 ISA circuit, the L = 6
  rehearsal truths the notebook computes openly, the transpiler's default layout, and the version stamps.
* **sealed (73)** — every statevector, the full and truncated Hamiltonian coefficient lists, `E0`,
  `E_adapt`, the VQE fidelities, the Trotter table, the Richardson error, the truncation shift, the
  gate counts and `dominant_error`. These are `savez_compressed`-ed into memory and XOR-ed with a
  seeded keystream, then stored as `_sealed_blob` + `_sealed_seed`.

`fallfest_grader._refs()` returns a `_RefMapping` that decodes the blob on first access, so every
existing `r["key"]` call is unchanged. **This is obfuscation, not cryptography, and the guide says
so**: it stops the answer key being readable in one line, which is the realistic failure mode for a
grader that must ship its own oracle. The defences that carry weight remain the hidden-backend
re-grade, `organizer/grade_submission.py` and the viva.

Verified: `np.load('reference_data/grader_refs.npz')` exposes none of
`E0_L8, H_full_coeffs_L8, sv_wave_L6, truncation_shift_L8_t4, trotter_table_t4_dt1, vqe_fid_L8,
n2q_logical_t8, dominant_error, gs_L8`, while the grader still reads all 119 keys.

## 3. Hardware metric change (ex 4.2)

Previously the RMSE term used only the finite window sites and a flat 5 %-per-NaN multiplier, so
NaN-ing the two worst sites *raised* the score (12.09 → 12.54). Both the points and the leaderboard
rank are now computed on the NaN-filled profile (a missing site contributes as "no signal"), and the
contrast/shape points require the six central sites to have been measured. Measured on the cached
ibm_kingston ODR set:

| NaN sites hidden | points | RMSE_W (diagnostic) | rank RMSE |
|---|---|---|---|
| 0 | 12.09 | 0.1253 | 0.1253 |
| 1 | 5.52 | 0.1124 | 0.1619 |
| 2 | 4.33 | 0.0958 | 0.1917 |
| 4 | 2.00 | 0.0861 | 0.2514 |

The headline calibration is unchanged: the cached ODR set still scores **12.09/18** before the 75 %
fallback cap, matching SPEC §E. `score_hardware` also pins the post-selection threshold to 0.01 when
it calls the team's `odr_mitigate`, so a team's own default cannot move the metric.
`organizer/grade_submission.py` ranks on `RANK-RMSE` and marks a submission with fewer than 16 finite
window sites as not ranked.

## 4. Tests

| command | result |
|---|---|
| `python organizer/make_grader_refs.py` | 46 public + 73 sealed keys, 3.1 MB, 25 s |
| `python tools/test_reference.py` | ALL TESTS PASSED; t = 8 bond-40 vs the QDC file nrmse 3.5e-5 |
| `python tools/test_grader.py` | ALL GRADER TESTS PASSED, 44 s (now 60+ variants incl. the new regressions) |
| `python tools/smoke_part_a.py` | 0 failures, 0 leaks, 86 s over 45 code cells |
| `python tools/smoke_part_b.py` | SMOKE TEST PASSED, 70 s over 27 code cells |
| `python tools/build_notebooks.py --execute` | see §5 |

New regression tests in `tools/test_grader.py`: a sign-flipped high-index observable (finding 4);
four inputs that used to raise (`measure_all` on a prep circuit, a `trotter_step` that ignores `L`,
1-element-list VQE values, a `None` wall time); numpy-typed `E0_L8` and string table keys; an
improvement run with no uncertainties; and an assertion that NaN-ing window sites can never improve
either the points or the rank.

`tools/smoke_part_b.py` needed two variables added to its preamble (`RUN_ON_HARDWARE`, `C_ref` and
the reference profile) because Part B now takes them from Part 0 rather than redefining them.

## 5. Execution and score

`python tools/build_notebooks.py --execute --timeout 5400` — **188 s, 0 error outputs**, 121 cells
(72 code, 66 with outputs). Slowest cells: 1.5 bond scan 33 s, B2 noise toy 33 s, 3.3 rehearsal 21 s,
2.3 grading 19 s, 2.3 error tables 16 s, 1.4 Trotter table 15 s. The whole notebook runs offline with
no IBM credentials.

Reference-solution score (`organizer/solution_score.json`):

| exercise | points | detail |
|---|---|---|
| `ex0.1` | 2.0 / 2 | chi_j = (-1)^j Z_j + I verified at L=5,7 |
| `ex0.2` | 2.0 / 2 | both 2-qubit rotations match their generators at theta=0.5030 |
| `ex0.3` | 2.0 / 2 | vacuum statevectors match at L=6,8 (fidelities 1.0000000000, 1.0000000000) |
| `ex0.4` | 2.0 / 2 | wavepacket statevectors match at L=6,8 (fidelities 1.0000000000, 1.0000000000) |
| `ex0.5` | 2.0 / 2 | 2nd-order step (ratios 7.09, 7.77), Fig.-8 ordering (F=1.0000000000), mitigation returns |
| `ex1.1` | 5.0 / 5 | H (identity excluded) and H_el^(1) match at L=4,8; [H,Q]=0, <Q^2>=0; E0(L=8) = -2.50901  |
| `ex1.2` | 4.0 / 4 | electric layer exact at L=4,6 (infidelities 2.2e-16, 4.4e-16); truncation shift 0.0101 c |
| `ex1.3` | 3.0 / 3 | fidelities 0.9961/0.9945/0.9929 and energy gaps confirmed at L=4,6,8 |
| `ex1.4` | 5.0 / 5 | Trotter table, dt^2 scaling, Richardson and dominant error all confirmed |
| `ex1.5` | 3.0 / 3 | bond convergence ok ({8: 0.04862326420240126, 12: 0.009697929553317497, 20: 0.0026746824 |
| `ex2.1` | 3.0 / 3 | 4 ISA circuits on one 68-qubit path [141, 142, 143]..[36, 41, 42] , 2q-depth 220, 68 lay |
| `ex2.2` | 5.0 / 5 | gate accounting (5758 logical 2q, 4958 CZ) and noise-matched mitigation circuit confirme |
| `ex2.3` | 5.0 / 5 | valid path on fake_kingston (15.0 s); cost 13.1294 vs baseline 13.2172 (ratio 0.993) |
| `ex2.4` | 2.0 / 2 | flight plan ok (predicted usage 31.5 s) and EstimatorOptions ok |
| `ex3.1` | 7.0 / 7 | ODR exact on synthetic CP-symmetric data (L=6,8,34), uncertainty within 25 % of MC, bias |
| `ex3.2` | 5.0 / 5 | twirled circuits ISA + equivalent, 96 % dressed CZs, 8 distinct seeds; charge post-selec |
| `ex3.3` | 8.0 / 8 | reduced noise model matches the target on chain [50, 51, 58, 71, 72, 73, 74, 75, 59, 55, |
| `ex4.1` | 2.2 / 3 | fallback dataset: capped at 75 % |
| `ex4.2` | 12.1 / 18 | RMSE_W 0.125 (5.0), contrast 0.437 (3.1), shape 2, quiet 0.021 (2), coverage 0.33, RMSE_ |
| `ex4.3` | 3.0 / 4 | RMSE_W 0.134 -> 0.052, z = 11.29, usage 182.0 s / usage_s = 182.0 is the organizer's est |
| `B1` | 3.0 / 3 | 16/32-step circuits and Richardson profile ok (organizer compares the arrays offline) |
| `B2` | 2.0 / 2 | toy ordering confirmed: {'raw_depol': 0.4661632728172159, 'odr_depol': 0.074441174928364 |
| `B3` | 1.0 / 1 | rzz barbell equals the CX barbell up to phase (6 rzz gates) |

**Autograded 82.4 / 90, bonus 6.0 / 6** (plus 10 judged = 100). Everything
through Part 3 is full marks; the Part-4 exercises are capped at 75 % because the solution runs on
the organizer fallback dataset rather than a live job, which is the intended behaviour
(4.2's 12.1/18 is exactly SPEC section E's calibration for the cached ibm_kingston ODR set).

## 5b. Packaging

`python tools/make_kits.py` writes two zips next to the challenge folder:

* `schwinger-hadron-challenge-participant.zip` — 20 files, 3.6 MB: the participant notebook,
  `fallfest_grader.py`, `challenge_utils.py`, `requirements.txt`, `README.md`, the QDC licence, the
  7 figures, and the reference data participants need (QDC t = 0 and bond-40 files, the bond-64/128
  MPS reference, the sealed `grader_refs.npz`). Excluded: the solution notebook, `organizer/`,
  `tools/`, `scratch/`, the cached ibm_kingston hardware set and its provenance, and
  `mps_reference_L34_dt.npz` (the Bonus B1 answer). The script asserts the participant zip contains
  no solution or organizer material.
* `schwinger-hadron-challenge-organizer.zip` — 51 files, 5.4 MB: everything except `scratch/`,
  `submission/` and `__pycache__`.

Verified by unpacking the participant zip into an empty directory: `ff.check_env()` passes, the
grader reads its sealed oracle, and the notebook runs to its first prompt with no errors.

## 6. Known limitations

* The sealed oracle is obfuscation, not security (§2).
* Function-level re-grading needs `submission/team_functions.py`, which the notebook's last cells now
  export automatically (2026-09-11); if a team deletes it or its import fails, the organizer tool keeps
  the local points provisionally, prints a `NOT RE-VERIFIED` line, and the viva covers those exercises.
* `select_chain`'s 60 s cap is enforced with a daemon thread: a runaway search is abandoned, not killed.
* The ex 2.3 cost ratio cannot by itself distinguish a real search from the transpiler's default
  layout on a well-calibrated device; the two judge flags exist for that.

## 7. Final pre-ship review pass (2026-09-11)

A five-lens review (participant experience, solution and physics, grader robustness, documentation,
organizer workflow), each finding reproduced independently before it was accepted. 47 findings were
reported; the ones acted on are listed below, the rest were minor wording items that were fixed in place
or refuted on reproduction.

**Participant-facing (notebook source `tools/nb_part_*.py`)**

* Exercise 4.3 crashed with `TypeError: 'NoneType' object is not subscriptable` in the kit as shipped
  (no fallback data → `improvement = None`, then an unguarded `np.savez`). The tail of the cell is now
  guarded and the job manifest records `{"strategy": None, "skipped": True}`; Part 5 and the bonus are
  reachable on Day 1 without any hardware data.
* The Part 4 banner promised that the notebook "loads the organizer's cached dataset" offline; the kit
  ships no such data. It now says what happens: the cells look for the Day-2 fallback files and
  otherwise skip with NaN placeholders.
* Rehearsal 3.3 ran the four PUBs through one `AerEstimatorV2` with a shared `seed_simulator`; Aer
  re-creates its precision-noise RNG per PUB, so the same "shot noise" landed on all four circuits and
  cancelled in wave − vacuum (ODR RMSE printed below the shot-noise floor). Each PUB now gets seed
  `7 + i`; prompt, theory text and the Part-4 checklist explain why.
* The viva protocol is now the same everywhere (8 questions from the 20-question bank in the 7-minute
  viva; the judges pick who answers; half credit 4–6, full ≥ 7): notebook Part 5, README, guide §6, and
  the new printable `organizer/judge_sheet.md`.
* The notebook's last cells export every notebook function and its constants to
  `submission/team_functions.py` and test-import the file, so the organizer re-grade can re-run the
  function-level exercises (see below). Listed in the README submission format, the notebook checklist,
  the guide §5 and SPEC §C.
* Wording: `chi_wave_exact` is computed in Part 0 (not "Part 1"); ex 2.2 asks for ≥ 99 % of the physics
  CZ count (not "within 1 %"); 2.3's search budget is 15 s (not "a few seconds"); the bond-64 MPS run is
  ~2 min (not ~6); the Rules box says "job or batch mode" and names the level-3 hint cost; the
  `schwinger_hamiltonian` scaffold now has the same `H = …` / `return H.simplify()` shape as its siblings;
  the checkpoint quotes both state-preparation numbers (1 − F ≈ 0.007, ≤ 0.04 in ⟨χ_j⟩) and the Part-5
  table carries both; the fallback canary prints that its FAIL verdict is informational.

**Grader (`fallfest_grader.py`)**

* The no-data stub (job id `none`, all-NaN evs) scored 1/3 on ex 4.1 for "job metadata" and printed the
  75 % fallback note; ex 4.1 and 4.2 now short-circuit to 0 with a "no hardware data" message.
* Wrong-shaped or wrong-typed *return values* (an `odr_mitigate` returning a dict or 69 values, a chain
  of `None`, an `X_raw` of the wrong length, a B2 entry of `None`, unbound `Parameter`s or classical
  bits in a returned circuit, numpy booleans in the estimator options) raised bare tracebacks; they now
  score 0 with a hint (`_as_vec`, `_as_int_list`, `_safe_statevector` / `_evolve_safe` everywhere a
  participant array or circuit is consumed). A non-finite `E0_L8` is a wrong answer, not a wrong type.
* NaN rule of the hardware metric: filling a hidden window site with 0 could still *gain* rank at the
  window edges (|X_ref| ≈ 0.01–0.06 there). A hidden site is now charged `max(|X_ref_j|, 0.25)` in the
  ranked RMSE and a hidden outside-window site counts as loud (1.0) in the quiet median. The notebook's
  `leaderboard_preview`, SPEC §E and the guide state the same rule; `test_grader.py` asserts that no
  single window site and no set of outside sites can be hidden for gain (the cached calibration, 18/18
  finite sites, is unchanged at 12.1/18).
* A corrupt `score.json` used to be silently replaced by an empty one; it is now moved aside with a
  warning. `check_env()` prints the submission directory as a relative path. A canary that is the exact
  t = 0 profile gets a judge flag. A real (non-fake) backend is graded against its own live calibration,
  and the ex 2.3 detail names the calibration source (`frozen:kingston` / `live:ibm_kingston`).
  Misleading hints fixed: a `trotter_step` that returns `None`, a missing `X_mit` (no longer a "JUDGE
  FLAG"), a non-finite `E0_L8`.

**Organizer tool (`organizer/grade_submission.py`, rewritten)**

* The guide documented four invocations of which three did not exist (zip input, `--leaderboard`,
  `--live`). The tool now accepts `<team>.zip`, the unzipped team folder or its `submission/` folder
  (several at once) and `--leaderboard CSV` writes a ranked CSV; `--live` is gone (the live job checks
  are a documented manual procedure).
* The re-grade used to copy the function-level scores (≈ 35 of 90 points) verbatim from the team's own
  `score.json`, while the guide claimed a notebook re-import. It now re-runs them from
  `team_functions.py`, prints a `NOT RE-VERIFIED` line naming every exercise it had to take on trust,
  cross-checks every `score.json` entry against the `[exN] p/m` lines in the executed notebook
  (`TAMPER?` flags), recomputes ex 4.2 with the team's exported `odr_mitigate` (QDC formula only as a
  fallback, and says which), really checks `truncation_shift` for ex 1.2, and collects every
  `FLAG` / `JUDGE FLAG` / `TAMPER?` into `organizer_score.json["flags"]`.
* Hidden mode is described as it is (a `select_chain` re-run on FakeFez/FakeMarrakesh); there is no
  second seed set, and the guide, README and SPEC no longer claim one.

**Documentation**

* Guide: real CLI and grading procedure (§5), automatic vs manual integrity checks, honest hidden-mode
  and oracle statements (`ff._refs()[key]` reads any key), corrected fallback-release steps (files live
  in `organizer/fallback_data/`, four files, what each exercise needs), a "Before the event" checklist,
  a complete regeneration table (`make_bonus_b1_reference`, `test_grader`, `smoke_part_*`, `make_kits`,
  `tools/trotter_ordering_check.py` — moved out of `scratch/` with its labels corrected) in the right
  order, the true participant-kit contents, and the stale numbers (dip −0.006, C_ref 0.514, 12/18,
  < 25 min, bond-64 110 s).
* SPEC: participant vs organizer kit contents, NaN rule, `grade_submission.py` and `team_functions.py`
  contract. README: garbled first sentence, QPU-time sentence, hint levels, submission-file lists
  (`circuits_isa.qpy`, `layout.json`, `flight_plan.json`, `rehearsal_L6.npz`, `team_functions.py`), honest
  re-grade description, fallback-dataset wording. Build notes: AI-session narrative and absolute
  home-directory paths removed from the organizer kit.

**Verification of this pass (2026-09-11, all on the pinned interpreter, `MPLBACKEND=Agg`)**

| check | result |
|---|---|
| `tools/test_grader.py` | ALL GRADER TESTS PASSED (59 s), including the new rows: no-data stubs for ex 4.1/4.2 → 0, single-site NaN sweep over every window site, robustness probes (unbound `Parameter`, ODR returning a dict/string, chain `None`, 68- and 69-entry ODR outputs, missing `X_mit`, numpy-bool options, exact t = 0 canary). |
| `tools/smoke_part_a.py` / `smoke_part_b.py` | both PASSED on the final source; Part B 79 s over 28 code cells; no `ff.grade_*` call leaks into a participant cell. |
| `tools/build_notebooks.py --execute` | participant 122 cells, solution 123 cells; solution executed in 210 s with 0 errors and 0 warnings; 82.4/90 autograded + 6/6 bonus (ex 4.1 2.25/3, ex 4.2 12.1/18, ex 4.3 3/4 — fallback dataset, capped at 75 %). `organizer/solution_score.json` is that run's `score.json`. |
| export cell | `submission/team_functions.py`: 65 functions + 240 constants, import OK, all graded functions present; the in-notebook self-check re-grades ex 0.1–0.5, 1.1, 1.2, 3.1, 3.2 and B3 from the exported module: all ok; no `__pycache__` left in `submission/`. |
| `organizer/grade_submission.py` | run on a fresh copy of that submission + executed notebook as a team folder, as `<team>.zip` and with `--hidden`: all exit 0 at 82.4/90 (+6); 85 callables imported; every function-level re-run passes; hidden ex 2.3 = 5/5 on FakeFez (cost ratio 1.023) and FakeMarrakesh (1.005); `NOT RE-VERIFIED: B2` only (toy-noise experiment, viva check); flags = the two fallback notices; `--leaderboard` CSV written and merged across runs. |
| tamper check | exercised earlier in the pass on a copy whose `score.json` had ex 2.4 and ex 4.3 raised by one point: both entries flagged `TAMPER? … but the notebook's last grader line shows …`. |
| no-data run | solution executed inside a participant-kit copy (no fallback files, `RUN_ON_HARDWARE = False`): 123 cells, 0 errors; ex 4.1/4.2 = 0 with the "no hardware data" message, ex 4.3 skipped cleanly, Part 5 and the bonus complete (B1 = 2/3 without the organizer arrays, by design). |
| kits | `tools/make_kits.py`: participant 20 files, organizer 51 files; the extracted participant zip has no organizer/tools/solution material, no `# KEEP` build tags, home paths or build-session text, no executed outputs, 35 `# PROMPT` blocks / 48 `YOUR CODE HERE` between the 49 `# BEGIN ANSWER` / `# END ANSWER` fill-in pairs (the participant convention explained in the introduction), and `ff.check_env()` passes from inside it with the oracle still sealed. |

Known limits left as they are (documented in the guide): hidden mode is a `select_chain` re-run only (no
second reference-seed set); the sealed oracle is obfuscation, not cryptography; the ODR threshold
pinning in the re-grade is best-effort; the tamper check covers only exercises whose `[exN] p/m` line
appears in the executed notebook; B2 is never re-run by the organizer tool.

## 8. Mitigation moved to the Runtime options (2026-09-11, client request)

Client requirement: no hand-rolled Pauli twirling; every error-mitigation or suppression technique other than
ODR goes through the Qiskit Runtime `EstimatorOptions`. Checked against the current docs (`TwirlingOptions`,
`ResilienceOptionsV2`, `DynamicalDecouplingOptions`, the local-testing-mode guide) and verified locally that
`qiskit_ibm_runtime.EstimatorV2` in local testing mode accepts a fully configured options object but applies
nothing except the shots (results bit-identical with and without twirling options).

* **Ex 3.2 (5 pts) replaced.** `twirl_circuit` / `postselect_charge` / the counts-based charge witness and the
  L = 4 twirl toy are gone. The exercise is now `mitigation_options(base_options, num_randomizations,
  shots_per_randomization, dd_sequence, readout_mitigation)`: a copy of the Part-2 options with gate + measurement
  twirling at the requested budget and a strategy, DD with the requested sequence, `resilience_level = 0` and no
  server-side mitigation on the ODR run, TREX (`resilience.measure_mitigation`) only when asked, and
  `max_execution_time` preserved. Every hardware job of Part 4 (canary, main run, improvement skeleton) now takes
  its options from this function (`with_twirling` removed). A check cell prints the option families, proves the
  Part-2 object is untouched, probes acceptance by the Runtime estimator in local mode and writes
  `submission/mitigation_options.json`. `grade_ex3_2(mitigation_options)` calls the function on a fresh
  `EstimatorOptions` base, on the same base with `readout_mitigation=True`, and on the dict form, and scores
  1 (copy, base untouched) + 1.5 (twirling) + 1 (DD) + 1.5 (resilience / max_execution_time). `_opt_get` maps the
  `Unset` sentinel to `None`; `_active_reduce` and `_blocks_before_cz` were removed.
* **4.3 menu**: the hand-twirled-PUB row is gone; ZNE is described as Runtime options; TREX and twirling/DD-budget
  rows added; a note says hand-rolled twirling or bitstring post-selection is not accepted. The TREX illustration
  is named as the `readout_mitigation=True` branch of 3.2, and the hardware skeleton shows an options-only run.
* **Bonus B2** is untwirled (exact density-matrix simulation cannot apply Runtime twirling): `toy_results` gains
  `witness_depol` / `witness_amp` and loses `odr_amp_twirl`; the second point is the charge-witness ordering
  (`witness_amp > 2 |witness_depol|` and `> 0.05`; reference run +0.040 vs +0.159).
* **Vocabulary**: the ODR per-site threshold is called the *retention threshold* everywhere participant-facing;
  "post-selection" is reserved for the (no longer allowed) bitstring filter.
* Updated in step: `tools/test_grader.py` (options fixtures, six ex 3.2 rows, B2 rows), `tools/smoke_part_b.py`
  (leak token), `organizer/grade_submission.py` (re-runs `mitigation_options`), the export self-check list, Part-A
  intro / rules / roadmap / 2.4 text, Part-3 header and 3.1/3.3 text, README, SPEC, ORGANIZER_GUIDE (rubric row,
  schedule, viva question 18, integrity note 4, B2 row), judge sheet.

**Verification** — `tools/test_grader.py` ALL GRADER TESTS PASSED (ex 3.2: reference 5/5; mutating the base,
dropping measurement twirling with resilience level 2, raising `max_execution_time`, TREX always on, returning
`None` all score below 5); `smoke_part_a` / `smoke_part_b` PASSED (no solution tokens in the participant view);
`build_notebooks.py --execute`: 122 / 123 cells, solution executed in 171 s with 0 errors and 0 warnings,
82.4/90 + 6/6 (ex 3.2 5/5, B2 2/2), export self-check all ok (60 functions); `organizer/grade_submission.py` on a fresh copy re-runs
`mitigation_options` from `team_functions.py` (ex 3.2 5/5) and reports the same 82.4/90 (+6), B2 the only
`NOT RE-VERIFIED` entry as before.
