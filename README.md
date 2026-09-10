# Hadron Dynamics in the Schwinger Model — Trust, but Verify

**UofT Qiskit Fall Fest 2026 · utility-scale challenge (68 qubits, real QPU, 100 points + 6 bonus)**

Based on *Quantum Simulations of Hadron Dynamics in the Schwinger Model using 112 Qubits*
(Farrell, Illa, Ciavarella, Savage, [arXiv:2401.08044](https://arxiv.org/abs/2401.08044)) and on the
IBM Quantum Developer Conference 2025 challenge of the same name
([qiskit-community/qdc-challenges-2025](https://github.com/qiskit-community/qdc-challenges-2025), Apache-2.0).

---

## What you will do

You will prepare a hadron wavepacket on a 68-qubit lattice (L = 34 spatial sites, staggered
fermions), evolve it to t = 8 with second-order Trotter circuits (about 5,000 CZ gates after
transpilation at optimization level 3, 5,300 at level 1), run it on an IBM Heron processor, and recover the vacuum-subtracted chiral condensate
profile X_j — a central dip with a peak on each side — using error mitigation you write yourself.

The twist compared with the original QDC notebook: before you are allowed to spend QPU time you must
*verify* every layer of the pipeline on your laptop, and most of the points are for those checks and
for what you do with the hardware data, not for filling in the original circuit code.

| Part | What | Points |
|---|---|---|
| 0 | Warm-up: the QDC building blocks (observables, R^XY gates, vacuum/wavepacket preparation, Trotter step) — graded by calling *your functions* at other lattice sizes | 10 |
| 1 | Physics you can verify: Hamiltonian vs exact diagonalization, the electric layer as an exact identity, quality of the pre-trained vacuum, Trotter error and Richardson extrapolation, MPS bond-dimension convergence | 20 |
| 2 | Engineering the 68-qubit experiment: one layout for four circuits, gate accounting and a noise-matched calibration circuit, calibration-aware qubit-chain selection, a flight plan with a QPU-usage prediction | 15 |
| 3 | Mitigation you wrote yourself: operator decoherence renormalization (ODR) with uncertainties and bias; the Runtime `EstimatorOptions` (twirling, dynamical decoupling, resilience) of every hardware job; a noisy rehearsal at L = 6 | 20 |
| 4 | Hardware: a t = 0 canary, the t = 8 main run (leaderboard), an improvement run with a z-score | 25 |
| 5 | Error budget and defence: a one-page error-budget table, an ODR bias derivation, an 8-minute pitch and a 7-minute viva (8 questions from a 20-question bank; the judges pick who answers) | 10 |
| Bonus | dt → 0 Richardson at L = 34, ODR on non-Pauli noise, a fractional-gate (`rzz`) barbell | +6 |

Everything up to Part 4 runs offline. Parts 0–2 take a typical team Day 1 and Part 3 usually spills
into Day 2 morning (the published schedule accounts for this); the canary and the main run need
about 40 seconds of QPU time, and the optional improvement run adds up to 60 s, all inside the 180 s cap.

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
  mps_reference_L34.npz (+ _meta.json)   organizer MPS reference, bond 64 and 128, t = 2, 4, 6, 8 (used for scoring)
  grader_refs.npz                    references used by the local grader (the answer keys inside are
                                     sealed: reading them out instead of doing the physics is cheating,
                                     and the organizer re-grade and the viva check for it)
  *_fallback.npz                     NOT in the kit: a real 68-qubit dataset (raw <Z_j>) that the organizers
                                     release on Day 2 to teams whose own job did not return; until then
                                     Part 4 prints a message and skips cleanly
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
* The organizers re-score your submitted arrays, re-run the functions the notebook exports to
  `submission/team_functions.py` (its last cells do this for you; `select_chain` is re-run on a
  different backend), recompute your hardware result from the raw expectation values with *your*
  `odr_mitigate`, and cross-check `score.json` against the grader lines in your executed notebook.
  Tampering with `submission/` or the grader is a disqualification.

### Submission format

Deadline and drop-off location are announced at kick-off. Hand in one zip named `<team>.zip`:

```
submission/                          everything the grader wrote (score.json, ex*.npz -- do not edit),
                                     plus the files the notebook saves:
                                       circuits_isa.qpy, layout.json, flight_plan.json   (2.1-2.4)
                                       mitigation_options.json                          (3.2)
                                       rehearsal_L6.npz                   (3.3)
                                       canary_t0.npz, canary_job_id.txt   (4.1, if you ran a canary)
                                       hardware_t8.npz, job_id.txt        (4.2)
                                       improvement.npz                    (4.3)
                                       job_info.json                      (every job id, written for you)
                                       error_budget.md, report.md (+ figures)   (Part 5)
                                       team_functions.py                  (your functions, exported for you)
schwinger_hadron_participant.ipynb   your notebook, executed, outputs included
```

`job_info.json` is written by the notebook and lists every IBM Quantum job id you used (canary, main,
improvement) with its backend and measured usage; the judges re-fetch these against your account.
The report is `submission/report.md`, **at most 2 pages including figures**: the error-budget table,
the ODR bias derivation and the rationale for your improvement run.

## Rules

1. **QPU budget: 180 seconds of usage per team, hard cap.** The notebook's flight-plan preflight
   must predict ≤ 180 s before any submission; after every job record `job.usage()`. Planned budget:
   canary ≈ 4 s, main run ≈ 32 s (4 PUBs × 64 twirls × 256 shots), improvement ≤ 60 s. Exceeding
   180 s costs 5 points; exceeding it deliberately voids Part 4.
2. **Teamwork.** Teams of 3–4. You may talk to other teams about physics and Qiskit, not exchange
   code or data. Mentors give the printed hints only (level 1 inline, level 2 released Day 1 evening,
   level 3 on Day 2 afternoon at the cost of 1 point on that exercise).
3. **Allowed resources.** The two papers ([arXiv:2401.08044](https://arxiv.org/abs/2401.08044),
   [arXiv:2308.04481](https://arxiv.org/abs/2308.04481)), the IBM Quantum documentation, the Qiskit API
   reference, the public QDC 2025 repository and any textbook. AI assistants may be used for Qiskit
   syntax; you must be able to explain every line in the viva.
   Copying the public QDC solution earns little by construction: its fill-ins are the 10-point
   warm-up, they are graded at other lattice sizes, the other 90 points do not exist in that notebook,
   and the viva asks you to explain your own results.

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
marked `# FF:`. The fallback dataset the organizers may release was recorded on `ibm_kingston` in July 2026. New
material is released under the same Apache-2.0 terms. Your submission remains your property.
