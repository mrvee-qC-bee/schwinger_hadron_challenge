# Build notes — grader (`fallfest_grader.py`, `organizer/make_grader_refs.py`, `organizer/grade_submission.py`, `tools/test_grader.py`)

Built 2026-09-07 on the organizer laptop (macOS, 8 cores), env `qiskit-paper`:
qiskit 2.5.2 / qiskit-ibm-runtime 0.49.0 / qiskit-aer 0.17.2 / numpy 2.3.0 / scipy 1.18.0.
All numbers below are printed by `organizer/make_grader_refs.py` (`scratch/make_grader_refs*.log`) or
measured by the scratch scripts named in each section (`scratch/*.py`, logs `scratch/*.log`; working
files, not part of the organizer kit).

## 0. The one thing to read first: the kinetic-layer ordering fix

SPEC E (draft) pinned the Trotter step to "Fig. 8: `Hkin(t/2)[even bonds, odd bonds] … Hkin(t/2)[odd, even]`" and
`schwinger_reference.trotter_step` implemented that.  The builder task in SPEC E asked to confirm by bond-40 MPS
that this ordering matches the QDC bond-40 file better than the alternative.  It does **not** — the opposite is true:

| kinetic ordering (first half-step / last half-step) | max \|X − X_QDC40\| | RMSE | nrmse | Pearson |
|---|---|---|---|---|
| **odd bonds first / even bonds first** (now pinned) | **1.99e-05** | 4.9e-06 | 0.0000 | 1.00000 |
| even bonds first / odd bonds first (old reference) | 2.34e-02 | 6.6e-03 | 0.0478 | 0.99871 |

(`scratch/ordering_check.py`, L=34, t=8, bond 40, seed 7, 29 s per ordering.)  A 2e-5 match of a truncated MPS
run is only possible with the identical gate sequence, i.e. the QDC file was produced with odd-first.  Two more
independent confirmations:

* `images/2ndTrott_circuits.png` zoomed (`scratch/fig8_topleft.png`, `fig8_topright.png`): the first orange
  column starts on wire 1 (bonds (1,2),(3,4),…), the second column on wire 0; the last two columns are the mirror
  image.  Fig. 8 **is** odd-first.
* The cached ibm_kingston run (`reference_data/hardware_ibm_kingston_2026-07-25.npz`) was produced by
  the organizer's benchmark code (`benchmarkSchwinger/schwinger_circuits.py`, not shipped): odd bonds
  first, then even; final half-step even then odd.
* The SPEC's own calibration numbers (X_ref centre 0.507/−0.006, RMSE_0 0.269, C_ref 0.515) are reproduced by the
  odd-first bond-64 reference (centre `[0.507 -0.006 -0.006 0.507]`), not by the even-first one (`[0.496 -0.002 …]`).

**Changes made (genuine bug in the source of truth, allowed by the task rules):**

1. `organizer/schwinger_reference.py::trotter_step` — swapped the two kinetic loops in both half-steps (odd
   first, then even; last half-step even then odd) and rewrote the docstring.  Backup of the old file:
   `scratch/schwinger_reference.py.bak`.
2. `organizer/SPEC.md` — corrected the ordering line in section E and the "0.023" fact in section D
   (both marked "[Corrected 2026-09-07 by the grader build]").
3. `reference_data/mps_reference_L34.npz` regenerated with `tools/make_mps_reference.py` (even-first copy kept
   at `scratch/mps_reference_L34_evenfirst.npz`).  Timings of the regeneration (odd-first): bond 64 t=2/4/6/8:
   13/57/78/110 s; bond 128 t=2/4/6/8: 24/327/489/606 s (total ≈ 28 min; `reference_data/make_mps_reference.log`).
4. `reference_data/grader_refs.npz` regenerated.

**Consequences for the other agents:** any solution cell that copied the old `trotter_step` must be updated
(the ex 0.5 grader now fails the even-first ordering with the hint "valid second-order step but not the Fig. 8
ordering (odd bonds (1,2),(3,4),.. first in the first kinetic half-step, even bonds first in the second)").
Numbers that changed: transpiled CZ count (O3 seed 42) 5278 → **4958** (consecutive steps now end/start with
the same sublattice, so the transpiler merges the two adjacent `RXXplus(dt/4)` layers), 2q-depth 238 → **220**,
the Trotter table (section 4) and the L=6/L=8 physics statevectors.  Hardware metric: the 0.023 ordering
difference is far below the hardware noise floor (RMSE_W 0.126), so the cached Kingston scores move by < 0.1 pt.

## 1. Reference data (`reference_data/grader_refs.npz`, 116 keys, 3.6 MB, 36 s to build)

Physics constants (odd-first ordering, m = 0.5, g = 0.3, +m/2·I per site):

| quantity | value |
|---|---|
| E0 (full H) L=4 / 6 / 8 | −1.1858741 / −1.8474360 / −2.5090053 |
| E_ADAPT L=4 / 6 / 8 | −1.1795258 / −1.8380457 / −2.4965655 |
| VQE fidelity L=4 / 6 / 8 | 0.9961026 / 0.9945246 / 0.9929329 |
| mitigation circuit return fidelity (L=6, t=2) | 1 − 1.3e-13 |
| truncation shift L=8, t=4 = max_j \|X_full − X_trunc\| | **0.010098** (L=6, t=4: 0.007144) |
| Trotter table L=8 (max_j \|X_trotter − X_exact,trunc\|) t=2: dt=1 / 0.5 / 0.25 | 0.20177 / 0.04991 / 0.01241 (ratios 4.04, 4.02) |
| same, t=4 | **0.09477 / 0.02405 / 0.00604** (ratios 3.94, 3.98) |
| Richardson (4X(0.25) − X(0.5))/3 error t=2 / t=4 | 9.2e-05 / 1.86e-04 |
| dominant_error | 'trotter' (0.095 > 0.010) |
| noiseless 4-step circuit vs exact truncated H at L=6, t=4 (ex 3.3 target `X_trotter_L6_t4`) | max dev 0.109 |
| n2q_logical (t=8, 8 steps) / 16 steps / 32 steps | **5758 / 10974 / 21406** (SPEC quoted 5756/10972/21404: the SPEC numbers are 2 short — same offset in all three, presumably the O_11 gate was not counted; the grader uses the measured values, ±1 %) |
| n_cz_physics FakeKingston seed 42: O3 (= O2) / O1 | **4958 / 5292** (SPEC quoted 5294 — that was the even-first O1-like count) |
| 2q-depth O3 | 220 |
| X_ref (bond 64, t=8) window constants | RMSE_0 = 0.2687, C_ref(measured) = 0.514 (SPEC 0.515 kept), max \|X64 − X128\| = 9.1e-04, max \|X_QDC40 − X128\| = 1.7e-03 |

Ordering-independent oracle used by ex 0.5: one Trotter step vs `expm(−i dt H_trunc)` at L=6 on 3 random states,
**phase-aligned** max-abs error (the circuit differs from the exponential by a global phase; without alignment
the ratios come out ≈ 2 for everything — `scratch/ratio_probe.py` vs `ratio_probe2.py`):

| step | e(0.4) / e(0.2) / e(0.1) | ratios |
|---|---|---|
| reference (odd-first) | 5.89e-4 / 8.31e-5 / 1.07e-5 | 7.09, 7.77 |
| even-first | 5.74e-4 / 7.53e-5 / 9.24e-6 | 7.62, 8.15 |
| first-order (no symmetrization) | 6.53e-3 / 1.78e-3 / 4.42e-4 | 3.67, 4.03 |
| wrong mass sign | 6.3e-2 / 3.1e-2 / 1.5e-2 | 2.0, 2.0 |

→ acceptance window [6, 10] separates second-order from first-order (≈4) and broken (≈2) steps; the pinned
ordering is then enforced by the fidelity with the stored L=6, t=2 statevector (even-first gives F = 0.987).

Frozen fake-backend summaries (runtime 0.49.0 `FakeKingston/FakeFez/FakeMarrakesh`): 176 undirected edges each;
CZ error median 0.0018 / 0.0039 / 0.0033; edges with error ≥ 0.5 (dead): 7 / 7 / 13; readout ≥ 0.5: 1 / 0 / 0;
qubits with `t1 = None`: 1 / 0 / 0 (stored as NaN).  Also stored: the grader-owned L=6, t=2 ISA circuit
(FakeKingston O1 seed 42, 264 CZ, layout 0..11) as QPY bytes, so the shipped grader contains no Trotter-step
solution code.

## 2. Tolerances and their rationale

| ex | check | tolerance | rationale / measurement |
|---|---|---|---|
| 0.1 | Pauli dict equality at L=5,7 | 1e-9 | exact construction |
| 0.2 | Operator vs expm at seeded θ=0.5030 | 1e-8 up to phase | exact; hints for sign / double-angle errors |
| 0.3/0.4 | statevector fidelity L=6,8 | > 1 − 1e-9 | exact (measured 1 − 1e-13) |
| 0.5 | ratio test / pinned fidelity / mitigation return | [6,10], > 1 − 1e-9, > 1 − 1e-9 | table above |
| 1.1 | Pauli coeffs (identity excluded) | 1e-9 | exact; E0 within 1e-4 (half credit for −6.50901 = no +m/2·I) |
| 1.2 | electric layer infidelity (random t ∈ [0.15, 0.6], 3 random states) | 1e-10 | measured 2e-16; truncation shift ±0.002 (value 0.0101; the prep_vacuum vs prep_vacuum_for_subtraction choice changes it by < 1e-4) |
| 1.3 | fidelity / gap | 1e-3 | truncated-H fidelities are 0.010 lower → caught |
| 1.4 | table entries | ±0.003 | dt=0.25 entries are 0.006–0.012, so 0.003 still requires the right ordering; ratios [3, 5.5] measured 3.94–4.04 |
| 1.5 | bond errors; X(40) vs X(64) | 8: > 0.02, 20: < 0.02; ±0.01 | `scratch/mps_calib.py` at t=8: bond 8: 0.0561, 20: 0.0029, 32: 0.0013, 40: 0.0007 (default and 1e-10 truncation threshold identical to 4 digits), bond 64 vs 128: 0.0009; QDC bond-40 file vs bond 64: 0.0008.  0.01 is 12× the bond-40 error and 2.3× **below** the ordering-swap signature (0.023), which gets its own hint |
| 2.1 | 2q-depth | [200, 300] | 220 at O3 (238 at O1) |
| 2.2 | counts | ±1 % | O1 count gets half credit with a hint; matched circuit: CZ(mitig) ≥ 0.99 CZ(phys) and per-bond \|Δ\| ≤ 2 (`scratch/perbond.py`: barrier-protected mitig = +2 on 34 bonds / 0 on 33; unprotected = −2/−4 on every bond, 4790 vs 4958 total) |
| 2.3 | cost ratio vs baseline | 1.05 / 1.15 / 1.30 | baseline = seeded randomized greedy DFS (`baseline_chain`, seed 2026, 2000 restarts, Gumbel temperature 1.0, 20000 expansions per restart, 10 s cap; 0.8–1.5 s on this laptop).  Costs: Kingston 1.1981, Fez 0.9969, Marrakesh 0.9409.  Same search with other seeds/4000 restarts: ratios 0.95–1.00 (Kingston), 1.002–1.004 (Fez), 1.000 (Marrakesh).  The transpiler's own O3 layout costs 1.863 (ratio 1.56 → 2/5).  The July hardware layout is invalid on the frozen target (dead edge (69,68) on Fez/Kingston) |
| 2.4 | usage model | 2 + 0.45e-3·4·twirls·shots ≤ 120 s, executions ≤ 160000, predicted within 25 %+1 s, σ_X ∈ (1e-3, 0.5) | SPEC QPU plan |
| 3.1 | ODR | 1e-9 (noise-free) + noisy replica (σ = 0.005): \|out − χ_true\| ≤ 8σ/f_pool, NaN where both partners dead | noise-free data cannot distinguish pooled from per-site ODR (dividing back is exact), so the synthetic set forces one fully dead mirror pair and one half-dead pair and adds a noisy replica; uncertainty within 25 % of a 1e4-sample MC at f > 0.2 (50 %: half credit); bias 1e-12 |
| 3.2 | twirl | ISA, same CZ list, state-equivalence on the 12 active qubits (2 random states, 1e-8, 4 seeds), ≥ 70 % dressed CZs, 8 distinct seeds | reference twirl: 96 % dressed |
| 3.3 | noise-model params | 1e-9 | parsed from `NoiseModel.to_dict()`: depolarizing p = (1 − P_I)·4ⁿ/(4ⁿ − 1), readout p = mean of the off-diagonal (asymmetry must be < 1e-9) |
| 3.3 | RMSE / factors | mit ≤ 0.8 raw and ≤ 0.10; factors ∈ (0.02, 0.97) on ≥ 8/12 | reference run: raw 0.0518 → mit 0.0131; factors 0.80–0.92; SPEC's (0.05, 0.95) left only 0.03 margin at the chain ends (0.922, 0.903), widened |
| 4.1 | canary | usage ≤ 30 s, wave Pearson ≥ 0.5, centre packet retention ≥ 0.05, flagged sites = retention < 0.4 (±2) | ibm_boston 2026-07-27 canary: Pearson 0.65, packet 0.08 (vacuum-arm Pearson is meaningless: flat profile) |
| 4.2 | metric | exactly SPEC E | see section 5 |
| 4.3 | z-score | within 20 % of grader's linear-propagation z | see code |

## 3. ex 3.3 recipe and timing (`scratch/ex33_probe.py`, `scratch/dm_timing.py`)

`reduced_noise_model(backend, chain)`: `NoiseModel(basis_gates=['cz','sx','x','rz','id'])`,
`depolarizing_error(eps_CZ, 2)` on `cz` for both directions of each chain edge, `depolarizing_error(eps_sx, 1)`
on `sx` and `x`, `ReadoutError([[1-p,p],[p,1-p]])` per chain qubit (values from `backend.target`).  Run the four
L=6, t=4 ISA circuits (O3, seed 42, `initial_layout=chain`, chain = the transpiler's own choice
`[52,53,54,55,59,75,74,73,79,93,94,95]`, 398/408 CZ) with
`AerEstimatorV2(options={"backend_options": {"noise_model": nm, "method": "statevector"}, "run_options": {"shots": 4000, "seed_simulator": 11}})`
and `obs.apply_layout(qc.layout)`.

* statevector, 4 PUBs × 4000 shots: **93–104 s** (per-shot trajectories; 400 shots: 5 s per 2 PUBs);
  `density_matrix`: 467 s (no truncation benefit) — do not use; `matrix_product_state`: 3× slower than statevector.
* Note: Aer's EstimatorV2 evaluates `save_expectation_value` on each trajectory, so `stds` come back as 0 and the
  readout part of the model is inert (no measurement instruction).  The grader therefore checks the readout
  parameters structurally only.  The notebook should say this explicitly.
* Result: RMSE vs the noiseless 4-step circuit: raw 0.0518 → ODR 0.0131 (ratio 0.25); ODR factors
  `[0.922 0.849 0.813 0.803 0.848 0.827 0.863 0.856 0.829 0.825 0.872 0.903]`.
* `tools/test_grader.py` caches the arrays in `scratch/ex33_cache_4000.npz` (`--no-cache` to redo).

## 4. Hardware metric on the cached data (`scratch/…` via `test_grader.py`)

Implemented exactly as SPEC E (window 25..42, `RMSE_W`, contrast at 31,32,35,36 vs 33,34, `quiet` = median of
\|X\| outside W, coverage with 2·sqrt(σ² + 0.003²), −5 % per NaN in W, X_hat must reproduce X_mit to 1e-6 (else 0 and a
judge flag), ODR factors < 0.9 on ≥ 10 window sites, usage > 180 s → −5, fallback cap 75 %).  Reference = odd-first
bond-64 profile.

| Kingston 2026-07-25 set (T=8) | RMSE_W | contrast | separation | quiet | points |
|---|---|---|---|---|---|
| odr strategy, ODR applied (R.odr_mitigate) | **0.126** | 0.437 | 0.379 | 0.021 | 5.0 + 3.1 + 2 + 2 = **12.1 / 18** |
| twirl_dd strategy, ODR applied | 0.135 | 0.421 | 0.383 | 0.025 | 11.5 / 18 |
| baseline strategy (no twirl/DD), ODR applied | 0.167 | 0.010 | −0.060 | 0.028 | 5.3 / 18 |
| odr set, raw X (no ODR) — a team submitting this fails the X_mit consistency gate anyway | 0.230 | 0.068 | – | 0.003 | 2.8 (metric only) |

Deviation from the SPEC estimate "≈ 10/18": the SPEC assumed `quiet ≈ 0.15 → 0 pts`; with the SPEC's own
definition (median over the 50 sites outside W) quiet = 0.021 (mean 0.031, **max 0.153** — that is where 0.15 came
from).  With the median the cached ODR set scores 12.1/18; the fallback cap (13.5) does not bite.  Retention in the
window 0.14–0.33, all 18 factors < 0.9.  Leaderboard extras: RMSE_68 = 0.076, coverage 0.33.

## 5. Test suite (`tools/test_grader.py`, ~35 s with the ex 3.3 cache, ~140 s without)

Perfect participant = `schwinger_reference` + wrappers written in the test (electric_layer, evolve_circuits_matched =
`protect_midpoint=True`, select_chain = randomized DFS with 4000 restarts, MC odr_uncertainty, odr_bias,
twirl_circuit with x/rz(π) Paulis and CZ-conjugated undo, postselect_charge = Hamming weight L, reduced_noise_model,
barbell_rzz via Walsh decomposition of the diagonal barbell → 6 rzz gates).  All autograded exercises score full
marks; ex 4.x are scored from the cached ibm_boston canary (3/3) and ibm_kingston T=8 sets (12.1/18; 4.3 gives 3.5/4
because twirl_dd → odr is not a significant improvement, z = 1.16).

Wrong variants that lose points with a hint (all in the table printed by the test): even-first ordering (1.5/2,
ordering hint), first-order step (0/2), prep_wave ignoring L (0/2), vacuum without O_11/O_22 (0/2), OV_3 interior
`range(2, L-1)` (0/2 at L=8), function that raises (0/2, message), H without the (−1)^k I part of Q_k (3/5, "single-Z
terms wrong"), truncated H as full (3/5), E0 without +m/2·I (4.5/5, convention hint), Rz layer wrong sign (0/4),
truncated-H fidelities (1.5/3), first-order Trotter table (0.3/5), even-first bond-40 profile (2/3 with the ordering
hint), non-monotone bond errors (0/3), 4th circuit on another layout (1/3), logical observables without apply_layout
(2/3), cancellable junction (3/5), O1 CZ count (4.5/5), transpiler layout as chain (2/5), hard-coded Kingston chain
on hidden FakeFez (0/5: dead edge), 67-qubit chain (0/5), oversized flight plan / resilience 2 (0/2), ODR without
post-selection (4/7), wrong bias ratio (5/7), y-gate twirl (1/5), no twirl (2.5/5), generic noise model with
inconsistent X_mit (2/8), tampered X_mit (0/18 + judge flag), 8-step circuit as 16 (2/3), wrong toy ordering (0/2),
CX barbell (0/1).

The final table is appended at the end of this file (section 8).

## 6. `organizer/grade_submission.py`

`python organizer/grade_submission.py <team_dir> [--hidden] [--team NAME] [--accept-quiet-note]` re-scores from
`score.json` + `exN.npz`, prints local-vs-organizer points, the leaderboard line and writes `organizer_score.json`.
Limitations (by construction — functions live in the notebook):

* function-level exercises (0.x, 1.1, 3.1, 3.2, the matched-circuit part of 2.2, the search of 2.3, the noise-model
  object of 3.3, the EstimatorOptions object of 2.4, 4.3, B1-B3) keep the local score unless the team exports
  `team_functions.py` into its submission folder; then they are re-run, and `--hidden` re-runs `select_chain` on
  FakeFez + FakeMarrakesh (mean of the two).  Without that file `--hidden` prints "ask the team to run
  select_chain(FakeFez()) at the viva".
* the hardware metric is recomputed with the QDC pooled ODR (`challenge_utils.postselection_and_mitigation`), which
  ex 3.1 forces the team's `odr_mitigate` to reproduce to 1e-9; the local X_mit consistency verdict is re-checked
  against it (both vacuum t=0 variants accepted for ex 3.3).

## 7. Open issues / decisions for the author

1. The ordering fix (section 0) must propagate to the notebook builder's solution cells and to any organizer text
   quoting 5278/5294 CZ or "0.023 vs QDC".  `tools/test_reference.py` [7] will now report ~2e-5 instead of 0.023.
2. `ex 4.2` quiet criterion: median (SPEC formula, implemented) vs max (SPEC's quoted 0.15).  Decide; the code has
   one line to change (`hardware_metric`).
3. ex 3.3 with Aer EstimatorV2: stds are 0 and readout errors are inert (section 3).  If the notebook wants
  readout effects and shot-noise stds it must use `SamplerV2` + own expectation values, or accept the limitation.
4. The ex 4.1 `flagged_sites` rule (vacuum-arm retention < 0.4) is the grader's; the organizer json for Boston used
   5σ + per-site retention on both arms and flagged 4 qubits (27, 63, 64, 154) vs the grader's 3 sites — ±2 tolerance
   covers it.
5. `select_chain` is run in a daemon thread with a 60 s join; a runaway search cannot be killed (Python), only
   abandoned.

## 8. Final `tools/test_grader.py` table (31 s with the ex 3.3 cache; exit 0, ALL GRADER TESTS PASSED)

```
exercise variant                                             pts  verdict
ex0.1   perfect                                        2.0/2    [ex0.1] 2.0/2 PASS chi_j = (-1)^j Z_j + I verified at L=5,7 [0.0s]
ex0.1   wrong sign (-1)^(j+1)                          0.0/2    [ex0.1] 0.0/2 -- hint: L=5: chi_0 differs from (-1)^j Z_j + I (max coeff diff 2.00e+00; remember Qiskit little-endian strings); L=7: chi_0 differs fro [0.0s]
ex0.2   perfect                                        2.0/2    [ex0.2] 2.0/2 PASS both 2-qubit rotations match their generators at theta=0.5030 [0.0s]
ex0.2   RXYminus with opposite sign                    1.0/2    [ex0.2] 1.0/2 -- hint: RXYminus(theta) != exp(+i theta/2 (XY-YX)) (max dev 9.64e-01) (looks like opposite sign) [0.0s]
ex0.3   perfect                                        2.0/2    [ex0.3] 2.0/2 PASS vacuum statevectors match at L=6,8 (fidelities 1.0000000000, 1.0000000000) [0.3s]
ex0.3   function raises                                0.0/2    [ex0.3] 0.0/2 -- hint: L=6: raised ValueError: boom (ValueError: boom); L=8: raised ValueError: boom (ValueError: boom) [0.0s]
ex0.3   OV_3 interior range(2, L-1)                    0.0/2    [ex0.3] 0.0/2 -- hint: L=6: fidelity with the reference vacuum = 0.998809; L=8: fidelity with the reference vacuum = 0.998809 [0.2s]
ex0.4   perfect                                        2.0/2    [ex0.4] 2.0/2 PASS wavepacket statevectors match at L=6,8 (fidelities 1.0000000000, 1.0000000000) [0.5s]
ex0.4   prep_wave ignores L                            1.0/2    [ex0.4] 1.0/2 -- hint: L=6: circuit has 16 qubits, expected 12 (does prep_wave ignore L?) [0.5s]
ex0.4   vacuum only                                    0.0/2    [ex0.4] 0.0/2 -- hint: L=6: this is the vacuum -- the O_11 / O_22 wavepacket layers are missing; L=8: this is the vacuum -- the O_11 / O_22 wavepacket [0.4s]
ex0.5   perfect                                        2.0/2    [ex0.5] 2.0/2 PASS 2nd-order step (ratios 7.09, 7.77), Fig.-8 ordering (F=1.0000000000), mitigation returns (F=1.0000000000) [0.4s]
ex0.5   even-first kinetic ordering                    1.5/2    [ex0.5] 1.5/2 -- hint: valid second-order step but not the Fig. 8 ordering (odd bonds (1,2),(3,4),.. first in the first kinetic half-step, even bonds  [0.4s]
ex0.5   first-order step                               0.0/2    [ex0.5] 0.0/2 -- hint: ratio test: error ratios 3.76, 4.05 ~ 4 -> first-order step (symmetrize: kinetic half-steps before AND after H_el, H_m); t=2 ph [0.3s]
ex1.1   perfect                                        5.0/5    [ex1.1] 5.0/5 PASS H (identity excluded) and H_el^(1) match at L=4,8; [H,Q]=0, <Q^2>=0; E0(L=8) = -2.50901 (convention: +m/2*I per site) [0.0s]
ex1.1   H_el without the (-1)^k I part of Q_k          3.0/5    [ex1.1] 3.0/5 -- hint: schwinger_hamiltonian(L=4): max coefficient difference 9.00e-02 (identity term excluded) (ZZ terms right, single-Z terms wrong: [0.0s]
ex1.1   E0 without +m/2 I                              4.5/5    [ex1.1] 4.5/5 -- hint: E0_L8 = -6.50901 drops the +m/2*I per site (16 sites x 0.25 = 4.0); the stated convention gives -2.50901 [0.0s]
ex1.1   truncated H passed as full                     3.0/5    [ex1.1] 3.0/5 -- hint: schwinger_hamiltonian(L=4): max coefficient difference 7.87e-02 (identity term excluded) (this is the TRUNCATED H_el^(1); the f [0.0s]
ex1.2   perfect                                        4.0/4    [ex1.2] 4.0/4 PASS electric layer exact at L=4,6 (infidelities 2.2e-16, 4.4e-16); truncation shift 0.0101 confirmed [0.0s]
ex1.2   perfect (array form)                           4.0/4    [ex1.2] 4.0/4 PASS electric layer exact at L=4,6 (infidelities 2.2e-16, 4.4e-16); truncation shift 0.0101 confirmed [0.0s]
ex1.2   Rz layer with wrong sign                       0.0/4    [ex1.2] 0.0/4 -- hint: electric_layer(L=4, t=0.171): infidelity 2.20e-03 vs exp(-i t H_el^(1)) (Rz layer: Rz(g^2 t) on 2k, Rz(g^2 t/2) on 2k+1 for k<L [0.0s]
ex1.3   perfect (live eigsh)                           3.0/3    [ex1.3] 3.0/3 PASS fidelities 0.9961/0.9945/0.9929 and energy gaps confirmed at L=4,6,8 [0.0s]
ex1.3   fidelity from truncated H                      1.5/3    [ex1.3] 1.5/3 -- hint: vqe_fidelity[4] = 0.98610, expected 0.99610 (overlap with eigsh ground state of the FULL H); vqe_fidelity[6] = 0.98452, expecte [0.0s]
ex1.4   perfect                                        5.0/5    [ex1.4] 5.0/5 PASS Trotter table, dt^2 scaling, Richardson and dominant error all confirmed [0.0s]
ex1.4   first-order table (ratio 2)                    0.3/5    [ex1.4] 0.3/5 -- hint: (2,0.5): 0.1000 vs reference 0.0499; (2,0.25): 0.0500 vs reference 0.0124; (4,1): 0.2000 vs reference 0.0948; (4,0.5): 0.1000 v [0.0s]
ex1.5   perfect (QDC bond-40 file)                     3.0/3    [ex1.5] 3.0/3 PASS bond convergence ok ({8: 0.0561, 20: 0.0029, 32: 0.0013, 40: 0.0007}); X(bond 40) within 0.0008 of the bond-64 reference; wall time [0.0s]
ex1.5   even-first bond-40 profile                     2.0/3    [ex1.5] 2.0/3 -- hint: max |X_bond40 - X_bond64| = 0.0241 > 0.01 (looks like the odd-first/even-first kinetic ordering was swapped: see ex0.5) [0.0s]
ex1.5   non-monotone errors                            1.0/3    [ex1.5] 1.0/3 -- hint: mps_err is not monotonically decreasing with the bond dimension: {8: 0.05, 20: 0.03, 40: 0.04}; expected err(bond 8) > 0.02 and [0.0s]
ex2.1   perfect                                        3.0/3    [ex2.1] 3.0/3 PASS 4 ISA circuits on one 68-qubit path [141, 142, 143]..[36, 41, 42] , 2q-depth 220, 68 layout-consistent observables [0.2s]
ex2.1   4th circuit on another layout                  1.0/3    [ex2.1] 1.0/3 -- hint: initial/final layouts differ between the 4 circuits (transpile all with initial_layout=<layout of the physics circuit>; a swap- [0.2s]
ex2.1   logical observables (no apply_layout)          2.0/3    [ex2.1] 2.0/3 -- hint: observable 0 != chi_0.apply_layout(layout) (use SparsePauliOp.apply_layout(qc_isa.layout)) [0.2s]
ex2.2   perfect                                        5.0/5    [ex2.2] 5.0/5 PASS gate accounting (5758 logical 2q, 4958 CZ) and noise-matched mitigation circuit confirmed [1.4s]
ex2.2   cancellable junction (no barrier)              3.0/5    [ex2.2] 3.0/5 -- hint: matched mitigation circuit: 4790 CZ vs 4958 in the physics circuit, worst per-bond difference 4 (the transpiler cancels the las [1.5s]
ex2.2   O1 CZ count                                    4.5/5    [ex2.2] 4.5/5 -- hint: n_cz_physics = 5292 is the optimization_level-1 count; level 3 merges the adjacent kinetic layers of consecutive steps -> 4958 [1.3s]
ex2.3   perfect (randomized DFS, 4000 restarts)        5.0/5    [ex2.3] 5.0/5 PASS valid path on fake_kingston (1.5 s); cost 1.1341 vs baseline 1.1981 (ratio 0.947) [2.2s]
ex2.3   returns 67 qubits                              0.0/5    [ex2.3] 0.0/5 -- hint: invalid chain: chain has 67 qubits, expected 68 [1.5s]
ex2.3   transpiler O3 layout as chain                  2.0/5    [ex2.3] 2.0/5 -- hint: valid path on fake_kingston (0.0 s); cost 1.8634 vs baseline 1.1981 (ratio 1.555) -- a calibration-aware search (randomized DFS [0.0s]
ex2.3   hard-coded Kingston chain, hidden FakeFez      0.0/5    [ex2.3] 0.0/5 -- hint: invalid chain: edge (69,68) has CZ error 1.000 >= 0.5 (dead edge) [0.1s]
ex2.3   perfect, hidden FakeFez                        5.0/5    [ex2.3] 5.0/5 PASS valid path on fake_fez (1.6 s); cost 1.0002 vs baseline 0.9969 (ratio 1.003) [2.6s]
ex2.4   perfect                                        2.0/2    [ex2.4] 2.0/2 PASS flight plan ok (predicted usage 31.5 s) and EstimatorOptions ok [0.0s]
ex2.4   dict options, resilience 2, too many shots     0.0/2    [ex2.4] 0.0/2 -- hint: predicted usage 362.0 s > 120 s (800000 executions); 800000 executions > 160000; predicted_usage_s = 30.0 but the usage model g [0.0s]
ex3.1   perfect                                        7.0/7    [ex3.1] 7.0/7 PASS ODR exact on synthetic CP-symmetric data (L=6,8,34), uncertainty within 25 % of MC, bias formula exact [0.6s]
ex3.1   ODR without post-selection                     4.0/7    [ex3.1] 4.0/7 -- hint: odr_mitigate(L=6): noise-free max deviation from chi_true 6.84e-14, NaN pattern wrong, noisy replica FAIL (sites where BOTH mir [0.6s]
ex3.1   bias with wrong ratio                          5.0/7    [ex3.1] 5.0/7 -- hint: odr_bias(chi_true, f_phys, f_cal) must equal (1 - chi_true) (1 - f_phys/f_cal) for arrays and scalars (value mismatch) [0.6s]
ex3.2   perfect                                        5.0/5    [ex3.2] 5.0/5 PASS twirled circuits ISA + equivalent, 96 % dressed CZs, 8 distinct seeds; charge post-selection ok [1.5s]
ex3.2   y gates (not ISA)                              1.0/5    [ex3.2] 1.0/5 -- hint: seed 0: 'y' on (1,) not supported by fake_kingston.target (express Paulis with x / rz(pi) / sx-based gates or re-run the transl [0.2s]
ex3.2   no twirl at all + weight L-1                   2.5/5    [ex3.2] 2.5/5 -- hint: 0 % of the CZs carry a non-identity frame (need >= 70 %) and < 8 distinct circuits for 8 seeds (draw a random 2-qubit Pauli per [1.1s]
ex3.3   perfect (Aer statevector, 4x4000 shots, 104s   8.0/8    [ex3.3] 8.0/8 PASS reduced noise model matches the target on chain [52, 53, 54, 55, 59, 75, 74, 73, 79, 93, 94, 95]; RMSE raw 0.0518 -> ODR 0.0131; fa [0.0s]
ex3.3   generic noise model, X_mit inconsistent        2.0/8    [ex3.3] 2.0/8 -- hint: noise_model: no cz error on (52,53); no cz error on (53,54); no cz error on (54,55) (+32 more); X_mit is not odr_mitigate(wave) [0.0s]
ex4.1   ibm_boston canary 2026-07-27 (20000 shots)     3.0/3    [ex4.1] 3.0/3 PASS canary ok: usage 18.0 s, median vacuum-arm retention 0.91, 3 flagged sites, verdict 'fail' [0.0s]
ex4.2   cached Kingston ODR set (T=8)                 12.1/18   [ex4.2] 12.1/18 -- hint: RMSE_W 0.125 (5.0), contrast 0.437 (3.1), shape 2, quiet 0.021 (2), coverage 0.33, RMSE_68 0.076, usage 96.0 s [0.0s]
ex4.2   same, marked fallback                         12.1/18   [ex4.2] 12.1/18 -- hint: RMSE_W 0.125 (5.0), contrast 0.437 (3.1), shape 2, quiet 0.021 (2), coverage 0.33, RMSE_68 0.076, usage 96.0 s | fallback dat [0.0s]
ex4.2   X_mit tampered (+0.1 in the window)            0.0/18   [ex4.2] 0.0/18 -- hint: not scored | JUDGE FLAG: submitted X_mit is not reproduced by the team's odr_mitigate from the submitted evs (1e-6) [0.0s]
ex4.2   raw twirl+DD set (T=8)                        11.5/18   [ex4.2] 11.5/18 -- hint: RMSE_W 0.134 (4.6), contrast 0.421 (2.9), shape 2, quiet 0.025 (2), coverage 0.22, RMSE_68 0.078, usage 96.0 s [0.0s]
ex4.3   Kingston twirl_dd -> odr                       3.5/4    [ex4.3] 3.5/4 -- hint: no significant improvement (z = 1.10); half credit for a documented null result [0.0s]
B1      perfect (structural)                           3.0/3    [B1] 3.0/3 PASS 16/32-step circuits and Richardson profile ok (organizer compares the arrays offline) [3.1s]
B1      8-step circuit passed as 16                    2.0/3    [B1] 2.0/3 -- hint: 16-step circuit has 5758 2q gates, expected 10974 +-1 % [2.7s]
B2      perfect ordering                               2.0/2    [B2] 2.0/2 PASS toy ordering confirmed: {'raw_depol': 0.12, 'odr_depol': 0.004, 'raw_amp': 0.1, 'odr_amp': 0.03, 'odr_amp_twirl': 0.006} [0.0s]
B2      wrong ordering                                 0.0/2    [B2] 0.0/2 -- hint: expected odr_depol < 0.25 raw_depol and odr_amp > odr_depol, got {'raw_depol': 0.12, 'odr_depol': 0.05, 'raw_amp': 0.1, 'odr_amp': [0.0s]
B3      perfect (rzz barbell)                          1.0/1    [B3] 1.0/1 PASS rzz barbell equals the CX barbell up to phase (6 rzz gates) [0.0s]
B3      cx barbell                                     0.0/1    [B3] 0.0/1 -- hint: circuit must use rzz gates and no cx/cz (ops: {'cx': 12, 'rz': 6}) [0.0s]
```

## Addendum (integration, 2026-09-07)

* **ex 2.3 cost definition changed** to the notebook's objective (expected number of errors on the t=8 circuit):
  `C = Σ_bonds 74·(−ln(1−ε_CZ)) + Σ_qubits [150·(−ln(1−ε_sx)) + (−ln(1−ε_RO))]` (`N_CZ_PER_BOND`, `N_SX_PER_QUBIT` in
  `fallfest_grader.py`; the baseline DFS noise is now scaled to the median bond+qubit cost). The old cost weighted one
  readout error like a single CZ, which made the grader disagree with what the notebook asks participants to minimise
  (solution notebook scored 2/5). New baselines: Kingston 13.217, Fez 22.519, Marrakesh 19.324; the notebook's 15 s search
  reaches 0.98–1.01×; the O3 transpiler layout on FakeKingston is 1.011× (so it is no longer penalised on cost — the
  hidden-backend call is what catches a hard-coded chain; the `test_grader` variant is now "reported" only).
* **ex 4.2 / 4.3 fallback**: the usage gates (−5 above 180 s; ≤ 60 s) are waived when the data is the organizer fallback
  (usage there is only an estimate); 4.3 is additionally capped at 75 % like 4.1/4.2. Four test variants added.
* See `organizer/BUILD_NOTES_integration.md` for the full list.
