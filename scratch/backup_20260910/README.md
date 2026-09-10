# Hadron Dynamics in the Schwinger Model — Trust, but Verify

**UofT Qiskit Fall Fest 2026 · utility-scale challenge (68 qubits, real QPU, 100 points + 6 bonus)**

Based on *Quantum Simulations of Hadron Dynamics in the Schwinger Model using 112 Qubits*
(Farrell, Illa, Ciavarella, Savage, [arXiv:2401.08044](https://arxiv.org/abs/2401.08044)) and on the
IBM Quantum Developer Conference 2025 challenge of the same name
([qiskit-community/qdc-challenges-2025](https://github.com/qiskit-community/qdc-challenges-2025), Apache-2.0).

---

## What you will do

You will prepare a hadron wavepacket on a 68-qubit lattice (L = 34 spatial sites, staggered
fermions), evolve it to t = 8 with second-order Trotter circuits (about 5,300 two-qubit gates after
transpilation), run it on an IBM Heron processor, and recover the vacuum-subtracted chiral condensate
profile X_j — a central dip with a peak on each side — using error mitigation you write yourself.

The twist compared with the original QDC notebook: before you are allowed to spend QPU time you must
*verify* every layer of the pipeline on your laptop, and most of the points are for those checks and
for what you do with the hardware data, not for filling in the original circuit code.

| Part | What | Points |
|---|---|---|
| 0 | Warm-up: the QDC building blocks (observables, R^XY gates, vacuum/wavepacket preparation, Trotter step) — graded by calling *your functions* at other lattice sizes | 10 |
| 1 | Physics you can verify: Hamiltonian vs exact diagonalization, the electric layer as an exact identity, quality of the pre-trained vacuum, Trotter error and Richardson extrapolation, MPS bond-dimension convergence | 20 |
| 2 | Engineering the 68-qubit experiment: one layout for four circuits, gate accounting and a noise-matched calibration circuit, calibration-aware qubit-chain selection, a flight plan with a QPU-usage prediction | 15 |
| 3 | Mitigation you wrote yourself: operator decoherence renormalization (ODR) with uncertainties and bias, Pauli twirling and charge post-selection, a noisy rehearsal at L = 6 | 20 |
| 4 | Hardware: a t = 0 canary, the t = 8 main run (leaderboard), an improvement run with a z-score | 25 |
| 5 | Error budget and defence: a one-page error-budget table, an ODR bias derivation, an 8-minute pitch and a short viva with the judges | 10 |
| Bonus | dt → 0 Richardson at L = 34, ODR on non-Pauli noise, a fractional-gate (`rzz`) barbell | +6 |

Everything up to Part 4 runs offline. Parts 0–3 take a typical team most of Day 1; the hardware runs
need about 40 seconds of QPU time in total.

## Prerequisites

* Python 3.11–3.13 and a laptop with ≥ 8 GB RAM (the L = 34 matrix-product-state simulations take
  10–40 s per circuit; nothing needs a GPU).
* Comfortable with `QuantumCircuit`, `SparsePauliOp`, the Qiskit transpiler
  (`generate_preset_pass_manager`) and the Runtime primitives (`EstimatorV2`, PUBs, `job.result()`).
* Some quantum many-body physics helps (Jordan–Wigner, Trotterization); the notebook explains the
  Schwinger-model specifics.
* An IBM Quantum account with an **Open Plan** instance (or the Classroom-Account instance the
  organizers give you) — see *Rules* below. Do not create a paid instance.

## Install

```bash
git clone <kit URL>   # or unzip the kit
cd schwinger-hadron-challenge
python -m venv .venv && source .venv/bin/activate        # or: conda create -n fallfest python=3.12
pip install -r requirements.txt                          # qiskit 2.5, qiskit-ibm-runtime 0.49, qiskit-aer 0.17, jupyter
jupyter lab schwinger_hadron_participant.ipynb
```

The first code cell runs `ff.check_env()` and prints the installed versions; it warns if you are below
Qiskit 2.5 / qiskit-ibm-runtime 0.49. Fake backends (`FakeKingston`, `FakeFez`, `FakeMarrakesh`) and
`qiskit-aer` are the only execution engines you need until Part 4.

Credentials: save your API key once with `QiskitRuntimeService.save_account(...)` on your own
machine (never paste a token into the notebook or commit it). The notebook only touches the service
inside cells guarded by `RUN_ON_HARDWARE = True`.

## Folder layout

```
schwinger_hadron_participant.ipynb   the challenge notebook (look for "# PROMPT" and "# YOUR CODE HERE")
challenge_utils.py                   given helpers: barbell, electric Trotter layer, RXXplus,
                                     QDC post-selection/ODR (reference only — you write your own), heavy-hex plots
fallfest_grader.py                   LOCAL autograder: ff.grade_ex0_1(...), ..., ff.summary()
reference_data/
  chi_*_t0_sim_L34.txt               exact t = 0 chiral condensate, L = 34 (from QDC)
  chi_*_evolved_sim_L34_maxbond40.txt   t = 8 bond-40 MPS profiles (from QDC)
  mps_reference_L34.npz              organizer MPS reference, bond 64 and 128, t = 2, 4, 6, 8 (used for scoring)
  grader_refs.npz                    small-lattice references used by the local grader
  hardware_ibm_kingston_2026-07-25.npz + _provenance.json   a real 68-qubit dataset (raw <Z_j>) to practise on
images/                              figures from the paper / QDC notebook
submission/                          created by the grader; this is what you hand in
requirements.txt, README.md, LICENSE-qdc-challenges-2025
```

## How the local grader works

Every exercise ends with a cell such as

```python
import fallfest_grader as ff
ff.grade_ex1_2(electric_layer, truncation_shift)    # -> "[ex1.2] 4.0/4 PASS — ..." or a hint
```

* Graders take your **functions** (or arrays) and run them at lattice sizes and angles you did not
  tune for, so hard-coded arrays or circuits copied at L = 34 do not pass.
* They never raise on a wrong answer, only on a wrong type; they print a short hint on failure and
  you can re-run them as often as you like (only the last result counts).
* Each call writes the points and a compact copy of your inputs to `submission/score.json` and
  `submission/exN.npz`. `ff.summary()` prints your current table.
* Nothing is sent anywhere: the grader is pure Python + numpy + Qiskit, no network.
* The organizers re-run the same graders on your submitted folder with a second, hidden set of seeds
  and lattice sizes, and recompute your hardware result from the raw expectation values with *your*
  `odr_mitigate`. Tampering with `submission/` or the grader is a disqualification.

### Submission format

Deadline and drop-off location are announced at kick-off. Hand in one zip named `<team>.zip`:

```
submission/                          everything the grader wrote, plus the hardware arrays it asks you to save:
                                     canary_result.npz, hardware_result.npz, improvement.npz, job_info.json
schwinger_hadron_participant.ipynb   your notebook, executed, outputs included
report.pdf                           at most 2 pages: error-budget table, ODR bias derivation, improvement rationale
```

`job_info.json` must list every IBM Quantum job id you used (canary, main, improvement), the backend,
the number of PUBs and the value of `job.usage()`; the judges re-fetch these.

## Rules

1. **QPU budget: 180 seconds of usage per team, hard cap.** The notebook's flight-plan preflight
   must predict ≤ 180 s before any submission; after every job record `job.usage()`. Planned budget:
   canary ≈ 4 s, main run ≈ 32 s (4 PUBs × 64 twirls × 256 shots), improvement ≤ 60 s. Exceeding
   180 s costs 5 points; exceeding it deliberately voids Part 4.
2. **Accounts.** Open Plan only (10 min per 28-day window per instance) on `ibm_fez`,
   `ibm_marrakesh` or `ibm_kingston`; job or batch mode — sessions are not available and must not be
   used. One instance per team for the scored runs; the four circuits of one run must be submitted as
   one job on one layout. Do not spread the PUBs of one run over several accounts.
3. **Hardware windows.** Jobs count only if submitted inside the announced windows (canary Day 1
   afternoon, main run Day 1 evening, improvement Day 2 midday) and before the hardware freeze on
   Day 2 afternoon. If your main job has not returned by Day 2 morning, ask the organizers for the
   fallback dataset (scored at 75 %).
4. **Teamwork.** Teams of 3–4. You may talk to other teams about physics and Qiskit, not exchange
   code or data. Mentors give the printed hints only (level 1 inline, level 2 released Day 1 evening).
5. **Allowed resources.** The two papers ([arXiv:2401.08044](https://arxiv.org/abs/2401.08044),
   [arXiv:2308.04481](https://arxiv.org/abs/2308.04481)), the IBM Quantum documentation, the Qiskit API
   reference, the public QDC 2025 repository and any textbook. AI assistants may be used for Qiskit
   syntax; you must be able to explain every line in the viva.
   Copying the public QDC solution earns little by construction: its fill-ins are the 10-point
   warm-up, they are graded at other lattice sizes, the other 90 points do not exist in that notebook,
   and the viva asks you to explain your own results.
6. **Integrity.** The grader recomputes your mitigated profile from the raw evs; submitted evs that
   coincide with the shipped practice dataset, jobs outside the windows, or an edited `submission/`
   folder are flagged and zero the exercise. Every team member must be present at the viva.

## Timeline (placeholder — the final schedule is announced at kick-off)

| | 3-day plan |
|---|---|
| Day 1 morning | kick-off, physics primer, Parts 0–1 |
| Day 1 afternoon | Part 2, **canary window** |
| Day 1 evening | Part 3.1–3.2, level-2 hints, **main-run window** (queue clears overnight) |
| Day 2 morning | Part 3.3, post-processing; fallback dataset for teams without a job |
| Day 2 midday | **improvement window**; hardware freeze mid-afternoon |
| Day 2 afternoon | Part 5 report, bonus |
| Day 3 | submission deadline (morning), pitches and vivas, leaderboard and awards |

## Links

* Paper: [arXiv:2401.08044](https://arxiv.org/abs/2401.08044) — hadron dynamics on 112 qubits;
  vacuum preparation: [arXiv:2308.04481](https://arxiv.org/abs/2308.04481) (SC-ADAPT-VQE).
* Original challenge: [qiskit-community/qdc-challenges-2025](https://github.com/qiskit-community/qdc-challenges-2025)
  (Track B, "Hadron dynamics in the Schwinger model").
* IBM Quantum docs:
  [Estimator options](https://quantum.cloud.ibm.com/docs/guides/estimator-options) ·
  [Error mitigation and suppression techniques](https://quantum.cloud.ibm.com/docs/guides/error-mitigation-and-suppression-techniques) ·
  [Configure noise management with Estimator](https://quantum.cloud.ibm.com/docs/guides/estimator-noise-management) ·
  [Fractional gates](https://quantum.cloud.ibm.com/docs/guides/fractional-gates) ·
  [Plans overview](https://quantum.cloud.ibm.com/docs/guides/plans-overview) ·
  [Classroom accounts](https://quantum.cloud.ibm.com/docs/guides/classroom-accounts) ·
  [Workload usage (estimate and query `job.usage()`)](https://quantum.cloud.ibm.com/docs/guides/estimate-job-run-time) ·
  [Local testing mode](https://quantum.cloud.ibm.com/docs/guides/local-testing-mode) ·
  [Transpile](https://quantum.cloud.ibm.com/docs/guides/transpile).
* Qiskit Aer MPS simulator: [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
  (`method="matrix_product_state"`, `matrix_product_state_max_bond_dimension`).

## Licence and attribution

`challenge_utils.py`, the four `reference_data/chi_*_L34*.txt` files and the figures in `images/` are
adapted from the QDC 2025 challenge (Apache-2.0, see `LICENSE-qdc-challenges-2025`); modifications are
marked `# FF:`. The cached `ibm_kingston` dataset was recorded by the organizers in July 2026. New
material is released under the same Apache-2.0 terms. Your submission remains your property.
