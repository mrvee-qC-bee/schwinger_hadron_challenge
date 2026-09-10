"""
Notebook source, part A: title, introduction, rules, roadmap, environment check,
Part 0 (warm-up), Part 1 (physics you can verify), Part 2 (engineering the 68-qubit experiment).

Exposes CELLS: an ordered list of nbkit md()/code() cells.  Part B (Parts 3-5, bonus) lives in
tools/nb_part_b.py; tools/build_notebooks.py concatenates the two lists.

Conventions (see tools/nbkit.py and organizer/SPEC.md section B):
    # PROMPT: ...            instruction, kept in both versions
    # BEGIN ANSWER/# END ANSWER   solution lines (stripped for participants; "# KEEP" lines survive)
    # PARTICIPANT: x = # description   +   x = <expr>  # SOL       QDC-style one-liners
All solution lines are copied from organizer/schwinger_reference.py (the notebook never imports it).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nbkit import md, code  # noqa: E402

CELLS: list[dict] = []
_add = CELLS.append

# =====================================================================================
# Title, attribution, introduction (adapted from the QDC 2025 notebook)
# =====================================================================================
_add(md(r"""
# Hadron Dynamics in the Schwinger Model — *Trust, but Verify*
### UofT Qiskit Fall Fest 2026 edition

Based on [*Quantum Simulations of Hadron Dynamics in the Schwinger Model using 112 Qubits* — arXiv:2401.08044](https://arxiv.org/abs/2401.08044)
and [*Scalable Circuits for Preparing Ground States on Digital Quantum Computers: The Schwinger Model Vacuum on 100 Qubits* — arXiv:2308.04481](https://arxiv.org/abs/2308.04481).

> **Attribution.** The physics narrative, the figures in `images/`, the helper module `challenge_utils.py` and the
> warm-up fill-ins of Part 0 are adapted from the IBM Quantum Developer Conference 2025 challenge
> *"Hadron dynamics in the Schwinger model"* (`qiskit-community/qdc-challenges-2025`, Apache-2.0; see
> `LICENSE-qdc-challenges-2025`). Everything from Part 1 onwards — the verification tasks, the engineering
> tasks, the hand-written ODR mitigation, the hardware protocol and the grader — is new material written for
> the Fall Fest and is graded on content that does **not** appear in the public QDC notebook.
"""))

_add(md(r"""
## Introduction

In this challenge you will simulate the propagation (spreading) of a hadron wavepacket in the Schwinger model —
quantum electrodynamics in one spatial dimension — on an IBM Quantum processor, and you will *verify every step*
before you spend a second of QPU time. The figure below (Fig. 10 of arXiv:2401.08044) shows the hadron evolution,
quantified through the chiral condensate $\mathcal{X}$ (left: classical simulation, right: hardware).

<img src="images/summary_ev.png" width="1100"/>

In the paper the wavepacket is prepared on a lattice with $L=56$ spatial sites ($2L=112$ staggered sites, i.e. 112 qubits)
and evolved up to $t=14$. As in the QDC challenge we use a smaller instance, $L=34$ (68 qubits), and a single evolved snapshot at $t=8$.
**The hardware goal is to reproduce the spread of the wavepacket at $t=8$: a dip of the vacuum-subtracted chiral condensate
at the centre of the lattice with a peak on either side** (simulated result for $L=34$ below).

<img src="images/wave_propagation_t8_sim.png" width="1100"/>
"""))

_add(md(r"""
## The Hamiltonian

We start from the lattice Schwinger model with $L$ spatial sites (Kogut–Susskind staggered fermions, $2L$ staggered sites,
open boundary conditions) and map it to $2L$ qubits with the Jordan–Wigner transformation. After eliminating the gauge field
with Gauss's law the Hamiltonian reads (Eqs. (1) and (6) of the paper)

$$
\begin{align}
\hat H & \ =\  \hat H_m + \hat H_{kin} + \hat H_{el} \ = \ \frac{m}{ 2}\sum_{j=0}^{2L-1}\ \left[ (-1)^j \hat Z_j + \hat{I} \right] \ + \ \frac{1}{2}\sum_{j=0}^{2L-2}\ \left( \hat \sigma^+_j \hat\sigma^-_{j+1} + {\rm h.c.} \right) \ + \ \frac{g^2}{ 2}\sum_{j=0}^{2L-2}\bigg (\sum_{k\leq j} \hat Q_k \bigg )^2
\ ,
\nonumber \\
\hat Q_k & \ = \ -\frac{1}{2}\left[ \hat Z_k + (-1)^k\hat{I} \right] \\
\hat{H}_{el}^{(Q=0)}(1)
\ = \ &\frac{g^2}{2}\Bigg\{ \sum_{n=0}^{\frac{L}{2}-1} \left[ \left( \frac{L}{2} - \frac{3}{4} - n \right)\hat{Z}_{2n}\hat{Z}_{2n+1}+\left (n+\frac{1}{4} \right )\hat{Z}_{L+2n}\hat{Z}_{L+2n+1}\right ]
\nonumber \\
&+ \ \frac{1}{2}
\sum_{n=1}^{\frac{L}{2}-2} \left (2 \hat{Z}_{2n} + \hat{Z}_{2n+1}-\hat{Z}_{L+2n}-2\hat{Z}_{L+2n+1} \right )
\nonumber \\
&+ \ \frac{1}{2}\left (2\hat{Z}_{0}+\hat{Z}_{1}+\hat{Z}_{L-2}-\hat{Z}_{L+1}-\hat{Z}_{2L-2}-2\hat{Z}_{2L-1} \right )  \nonumber \\
&+ \sum_{n=0}^{\frac{L}{2}-2} \bigg[\left (\frac{L}{2}-\frac{5}{4}-n \right )(\hat{Z}_{2n}+\hat{Z}_{2n+1})\hat{Z}_{2n+2} + \left( \frac{L}{2} - \frac{7}{4} - n \right)(\hat{Z}_{2n}+\hat{Z}_{2n+1})\hat{Z}_{2n+3}
\nonumber \\
&+ \  \left (n+\frac{1}{4} \right )(\hat{Z}_{L+2n+2}+\hat{Z}_{L+2n+3})\hat{Z}_{L+2n} +\left (n+\frac{3}{4} \right )(\hat{Z}_{L+2n+2}+\hat{Z}_{L+2n+3})\hat{Z}_{L+2n+1} \bigg] \Bigg\}
\ .
\end{align}
$$

$\hat H_{el}^{(Q=0)}(1)$ is the electric interaction *restricted to the charge-zero sector and truncated to range 1*
(the confining interaction is exponentially screened, see Sec. II of the paper); it is what the Trotter circuits implement,
while the first line is the full, untruncated model. **Part 1 asks you to build both and to measure what the truncation costs.**

### Conventions used everywhere in this notebook

* **qubit $j$ $\equiv$ staggered site $j$**, $j = 0,\dots,2L-1$. Even sites carry electrons, odd sites positrons.
* **Qiskit is little-endian**: in a Pauli *string* of length $2L$ the character for site $j$ is at position $2L-1-j$
  (the rightmost character is qubit 0). `SparsePauliOp.from_sparse_list([("Z", [j], 1.0)], 2L)` avoids the bookkeeping.
* $\hat H_m$ **includes the $+\tfrac{m}{2}\hat I$ per site**, so all energies are shifted by $+mL$ relative to the
  convention without it; the grader states which one it expects wherever an energy is graded, and
  tells you if your number looks like the other convention. Reported energies must follow the stated convention; the grader
  compares Pauli coefficients with the identity term excluded, so both Hamiltonian conventions pass the structural checks.
* Model parameters are fixed by the pre-trained SC-ADAPT-VQE angles: $m = 0.5$, $g = 0.3$.

### Electrons and positrons on the staggered lattice

For the (even) electron sites: qubit $|0\rangle$ = site occupied, $|1\rangle$ = site empty.
For the (odd) positron sites: qubit $|0\rangle$ = site empty, $|1\rangle$ = site occupied.
The strong-coupling vacuum (all sites empty) is therefore $|1010\cdots10\rangle$ in site order, i.e. an $X$ on every even qubit.

### Observables

The observable of interest is the chiral condensate $\hat{\chi}_j = (-1)^j \hat Z_j + \hat{I}$. The final quantity is the
**vacuum-subtracted chiral condensate** $\mathcal{X}_j \equiv \langle\hat{\chi}_j\rangle^{\rm wave} - \langle\hat{\chi}_j\rangle^{\rm vacuum}$,
the difference between the evolved wavepacket and the evolved vacuum. We therefore build separate circuits for the wavepacket
(initial state + evolution) and for the vacuum (initial state + evolution).

### Initial states, evolution, error mitigation

* Initial states come from the variational circuits trained in the paper with SC-ADAPT-VQE (2 steps for the vacuum, 2 steps for
  the wavepacket); we do not re-train them.
* Time evolution uses a second-order Suzuki–Trotter product formula (Fig. 8 of the paper).
* Error mitigation follows the paper. Pauli twirling and dynamical decoupling are supplied by Qiskit Runtime through the
  `EstimatorOptions` you build (2.4, 3.2); *operator decoherence renormalization* (ODR) with a calibration circuit per physics
  circuit, a per-site retention threshold on the calibration signal, CP (mirror) symmetrisation and vacuum subtraction are the
  physics of this challenge and you implement them yourself (Part 3).

### Summary of circuits needed

For the single snapshot $t=8$ we need four circuits, which must run on the **same qubits** with the **same noise**:
1. wavepacket physics circuit, 2. wavepacket calibration (mitigation) circuit, 3. vacuum physics circuit, 4. vacuum calibration circuit.
"""))

_add(md(r"""
<div class="alert alert-block alert-warning">

## What is different from the public QDC notebook (and why copying it will not score)

The QDC 2025 notebook is public, and so is a solved copy of it. We assume you have read it. This edition is built so that the
public solution gives you at most the 10 warm-up points:

* **Function-level grading at other lattice sizes.** Every circuit-building exercise is graded by *calling your function* at a
  smaller $L$ (and, where relevant, at a random angle) and comparing state fingerprints. Pasting a 68-qubit circuit does nothing.
* **Physics you can verify (Part 1, 20 pts).** You build the full and the truncated Hamiltonian, diagonalise them, check the
  electric Trotter layer against $e^{-it\hat H_{el}^{(1)}}$, measure the SC-ADAPT-VQE fidelity, the Trotter error, the truncation
  shift and the MPS bond-dimension convergence. None of this is in the QDC notebook — and it produces the error budget you defend in Part 5.
* **Engineering (Part 2, 15 pts).** A *noise-matched* calibration circuit (the QDC one silently loses ~170 CZ gates to the transpiler),
  a calibration-aware 68-qubit chain search on `backend.target`, and a QPU *flight plan* under a hard cap of **180 s** (the QDC run used
  6 min per job with 480 twirls x 400 shots).
* **Mitigation you wrote yourself (Part 3, 20 pts).** ODR, its uncertainty, its bias and the per-site retention threshold are
  written by you and rehearsed on a reduced noise model before touching hardware; twirling, dynamical decoupling and any other
  suppression or mitigation go through the Runtime `EstimatorOptions` (no hand-rolled Pauli frames, no bitstring post-selection). `challenge_utils.postselection_and_mitigation` is only a cross-check.
* **Hardware (Part 4, 25 pts).** Canary run at $t=0$, main run at $t=8$, improvement run — scored by a fixed metric against an
  organizer **bond-64 MPS reference** (`reference_data/mps_reference_L34.npz`, validated against bond 128), with a fixed scoring window
  and contrast. The grader recomputes your mitigated profile from your raw expectation values with your own `odr_mitigate`.
* **Pinned Trotter ordering.** Kinetic (odd bonds, then even bonds) - electric - mass - kinetic (even, then odd), exactly Fig. 8. The hardware
  reference is computed for this ordering; any other valid ordering passes the ratio test but fails the fidelity check.
* **Error budget + viva (Part 5, 10 pts).** One page, one derivation, eight minutes, questions.

</div>
"""))

_add(md(r"""
<div class="alert alert-block alert-success">

## Rules

1. **QPU budget: 180 s of usage per team, total, hard cap.** Every hardware job must be submitted from this notebook with the
   options of Part 2.4 (`max_execution_time = 180`); the flight plan (canary <= 30 s, main run ~32 s, improvement run <= 60 s)
   leaves a margin for one retry. Usage above 180 s costs 5 points; a job that could not return is scored on the
   organizer-released fallback dataset, capped at 75 %.
2. **Open Plan, job or batch mode.** No sessions, no `max_execution_time > 180`, no `resilience_level > 0` and no server-side
   mitigation on the main run (ODR needs the raw signal; Runtime TREX/ZNE are welcome in the 4.3 improvement run). Twirling and every
   mitigation other than ODR go through `EstimatorOptions` — hand-rolled twirling or bitstring post-selection is not accepted.
   Never put credentials in the notebook; `RUN_ON_HARDWARE = False` keeps every hardware cell inert.
3. **Submission = the `submission/` folder** written by the grader cells (`score.json`, `exN.npz`, `circuits_isa.qpy`,
   hardware results and job ids) plus this notebook, executed. Run `ff.summary()` at the end.
4. **Local grader.** `fallfest_grader.py` runs offline; it never raises on a wrong answer, prints a hint, and stores a compact copy
   of your inputs so the organizers can re-grade. Re-running a grader cell overwrites the previous score for that exercise.
5. **Viva.** Part 5 is an 8-minute pitch of your error budget plus questions from the judges. Every team member should be able to
   explain every graded number.
6. Work as a team; do not share code across teams. Anything reused from the public QDC notebook must be attributed (it is, in Part 0).

</div>
"""))

_add(md(r"""
## Roadmap and points (100 + bonus)

| Part | id | what you deliver | pts |
|---|---|---|---|
| 0 Warm-up (QDC fill-ins, graded at other $L$) | 0.1 | `chiral_condensate_observables(L)` | 2 |
| | 0.2 | `RXYplus(theta)`, `RXYminus(theta)` | 2 |
| | 0.3 | `prep_strong_coupling_vacuum`, `vacuum_prep_rotate_OV_3`, `prep_vacuum` | 2 |
| | 0.4 | `wave_prep_rotate_O_22`, `prep_wave` | 2 |
| | 0.5 | `trotter_step`, `evolve_circuits` | 2 |
| 1 Physics you can verify | 1.1 | Hamiltonian builders, charge-sector checks, `E0_L8` | 5 |
| | 1.2 | `electric_layer` identity check + `truncation_shift` | 4 |
| | 1.3 | `vqe_fidelity`, `vqe_energy_gap` | 3 |
| | 1.4 | Trotter error table, ratios, Richardson, `dominant_error` | 5 |
| | 1.5 | MPS bond-dimension convergence at $L=34$, $t=8$ | 3 |
| 2 Engineering the 68-qubit experiment | 2.1 | four ISA circuits + observables on one layout | 3 |
| | 2.2 | gate accounting + noise-matched calibration circuit | 5 |
| | 2.3 | `select_chain` calibration-aware path search | 5 |
| | 2.4 | `flight_plan` + `make_estimator_options` + usage prediction | 2 |
| 3 Mitigation you wrote yourself | 3.1 | `odr_mitigate`, `odr_uncertainty`, `odr_bias` | 7 |
| | 3.2 | `mitigation_options`: Runtime twirling, DD and resilience options of every hardware job | 5 |
| | 3.3 | noisy rehearsal at $L=6$, $t=4$ with your own ODR | 8 |
| 4 Hardware | 4.1 | canary at $t=0$ | 3 |
| | 4.2 | main run at $t=8$ (leaderboard metric) | 18 |
| | 4.3 | improvement run with z-score | 4 |
| 5 Error budget + defence | 5 | error-budget table, ODR bias derivation, pitch, viva | 10 |
| Bonus (max +6) | B1–B3 | Richardson at $L=34$, ODR on damping vs depolarising, fractional-gate barbell | 6 |

Every exercise follows the same pattern: a short theory block, a code cell with `# PROMPT:` and `# BEGIN ANSWER` / `# END ANSWER`
markers (fill in between the markers), and a grader cell. Level-1 hints are inline; level-2 hints are released by the organizers on Day 1 evening; a level-3 hint (Day 2 afternoon,
mentors only) costs 1 point on that exercise.
"""))

# =====================================================================================
# Environment check and setup
# =====================================================================================
_add(md(r"""
## Environment check

Target stack: `qiskit ~= 2.5`, `qiskit-ibm-runtime ~= 0.49`, `qiskit-aer ~= 0.17` (see `requirements.txt`). The grader is a local
module in this folder; it never contacts the network.
"""))

_add(code(r'''
import fallfest_grader as ff

ff.check_env()
'''))

_add(code(r'''
import os
import time
import math
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:   # keep figures inline inside Jupyter even when MPLBACKEND=Agg is set by a batch executor
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass
sns.set()
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("lines", linewidth=2)
plt.rc("font", size=12)
plt.rc("legend", fontsize="medium")
plt.rc("axes", labelsize=14)
plt.rcParams["figure.figsize"] = 15, 4

from qiskit import QuantumCircuit, qpy
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator

# helper functions shared with the QDC challenge (module located in the same folder as the notebook)
import challenge_utils
from challenge_utils import RXXplus, trotter_step_electric_2q

os.makedirs("submission", exist_ok=True)   # everything the grader and the organizers need ends up here
REF_DIR = "reference_data"

RUN_ON_HARDWARE = False   # every cell that would submit a job is guarded by this flag
RUN_BOND_64 = False       # Part 1.5: also run the bond-64 MPS yourself (~2 min on 8 cores); the organizer arrays are loaded otherwise
'''))

_add(md(r"""
### Model parameters [no prompts]

The variational angles used below were trained for $m = 0.5$, $g = 0.3$ — do not change them. $L$ is the number of *spatial*
sites; the staggered electrons and positrons occupy $2L$ sites, i.e. $2L$ qubits.
"""))

_add(code(r'''
# DO NOT MODIFY (the variational circuit parameters used in the rest of this challenge are specific to these model parameters)
m = 0.5   # electron/positron mass
g = 0.3   # coupling
L = 34    # number of spatial lattice sites, to be captured by 2 x L = 68 qubits
'''))

# =====================================================================================
# Part 0 — warm-up
# =====================================================================================
_add(md(r"""
<span id="part0"></span>
# Part 0 — Warm-up: the QDC circuits, graded at other lattice sizes (10 pts)

<div class="alert alert-block alert-success">

This part reproduces the circuit construction of the QDC notebook: observables, the two-qubit building blocks, the SC-ADAPT-VQE
vacuum and wavepacket circuits, and the second-order Trotter step. Every function is graded by calling it at $L \in \{4, 6, 8\}$
(and random angles), so write functions of `L`, never hard-code 68 qubits. By the end of Part 0 you will have the four $t=8$
circuits, exact $t=0$ condensates from an MPS simulation, and the $t=8$ reference profile.

</div>
"""))

# ---- 0.1 observables -----------------------------------------------------------
_add(md(r"""
### Exercise 0.1 — Chiral condensate observables (2 pts)

$\hat{\chi}_j = (-1)^j \hat Z_j + \hat{I}$ for $j = 0, \dots, 2L-1$. Return a list of $2L$ `SparsePauliOp`s, one per site, each a weighted
sum of two Pauli strings on $2L$ qubits.

*Hint (level 1):* remember little-endian — site $j$ is the character at position $2L-1-j$ of the string, or use
`SparsePauliOp.from_sparse_list`. Check: `observables[1]` must contain `-1.0 * Z` on qubit 1.
"""))

_add(code(r'''
# PROMPT: Complete the function so that it returns the list [chi_0, chi_1, ..., chi_{2L-1}] of SparsePauliOps.
def chiral_condensate_observables(L: int) -> list[SparsePauliOp]:
    """chi_j = (-1)^j Z_j + I for j = 0 .. 2L-1 (list of 2L SparsePauliOps on 2L qubits)."""
    n = 2 * L
    obs = []
    # BEGIN ANSWER
    for j in range(n):
        z = ["I"] * n
        z[n - 1 - j] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(z), (-1) ** j), ("I" * n, 1.0)]))
    # END ANSWER
    return obs


observables = chiral_condensate_observables(L)
print(len(observables), "observables;  chi_0 =", observables[0], "\n                 chi_1 =", observables[1])
'''))

_add(code(r'''
# grade your answer:
ff.grade_ex0_1(chiral_condensate_observables)
'''))

# ---- 0.2 RXY gates ---------------------------------------------------------------
_add(md(r"""
### Exercise 0.2 — The two-qubit building blocks $R^{(XY)}_{\pm}(\theta)$ (2 pts)

The variational layers are products of unitaries $e^{i\theta \hat O}$ generated by the pool operators of arXiv:2401.08044 (Eq. (7) and (12)):

$$
\hat O_{mh}^{V}(1) = \frac{1}{2}\sum_{n=0}^{2L-2}(-1)^n\left(\hat X_n \hat Y_{n+1} - \hat Y_n \hat X_{n+1}\right), \qquad
\hat O_{mh}^{V}(3) = \frac{1}{2}\sum_{n=0}^{2L-4}(-1)^n\left(\hat X_n \hat Z_{n+1} \hat Z_{n+2} \hat Y_{n+3} - \hat Y_n \hat Z_{n+1} \hat Z_{n+2} \hat X_{n+3}\right).
$$

All of them are built from two-qubit rotations (Fig. 5 of the paper; called $R_\pm(\theta)$ in arXiv:2308.04481, Fig. 3a):

<img src="images/xy_circ.png" width="700"/>

With the sign conventions of the figure these implement (verified numerically — you can check it in the next cell)

$$
R^{(XY)}_{+}(\theta) = \exp\!\left[-i\tfrac{\theta}{2}\left(\hat X\hat Y + \hat Y\hat X\right)\right], \qquad
R^{(XY)}_{-}(\theta) = \exp\!\left[+i\tfrac{\theta}{2}\left(\hat X\hat Y - \hat Y\hat X\right)\right] = e^{i\theta \hat O}, \quad \hat O = \tfrac{1}{2}(\hat X\hat Y - \hat Y\hat X),
$$

where the first Pauli acts on the first circuit qubit. Build each as a `QuantumCircuit(2)` and convert it with `to_gate` (no barriers inside a gate).

*Hint (level 1):* the only difference between the two gates is the sign of the `ry` rotation on qubit 0.
"""))

_add(code(r'''
# PROMPT: Complete the functions below so that they output a gate implementing R^{(XY)}_{\pm}(\theta) on two qubits (Fig. 5).
# NOTE: Do not add barriers to the qc because this will throw an error when converting the circuit to a gate object.
def RXYplus(theta: float) -> Gate:
    """R^{(XY)}_+(theta) = exp(-i theta/2 (XY + YX))."""
    qc = QuantumCircuit(2)   # build the gate in this circuit: the line below converts it with qc.to_gate()  # KEEP
    # BEGIN ANSWER
    qc.z(1); qc.h(1); qc.s(1)
    qc.s(0); qc.h(0)
    qc.cx(0, 1)
    qc.ry(theta, 0)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0); qc.sdg(0)
    qc.sdg(1); qc.h(1); qc.z(1)
    # END ANSWER
    gate = qc.to_gate(label=rf"$R^{{XY}}_{{+}}({theta:.4g})$")
    return gate


def RXYminus(theta: float) -> Gate:
    """R^{(XY)}_-(theta) = exp(+i theta/2 (XY - YX)) = exp(i theta O) with O = (XY - YX)/2."""
    qc = QuantumCircuit(2)   # build the gate in this circuit: the line below converts it with qc.to_gate()  # KEEP
    # BEGIN ANSWER
    qc.z(1); qc.h(1); qc.s(1)
    qc.s(0); qc.h(0)
    qc.cx(0, 1)
    qc.ry(-theta, 0)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0); qc.sdg(0)
    qc.sdg(1); qc.h(1); qc.z(1)
    # END ANSWER
    gate = qc.to_gate(label=rf"$R^{{XY}}_{{-}}({theta:.4g})$")
    return gate
'''))

_add(md(r"""
**Verify before you grade.** The cell below compares your gates with the matrix exponentials of the generators (up to a global phase).
`Operator` on a 2-qubit gate is cheap; this is the pattern you will use throughout Part 1.
"""))

_add(code(r'''
from scipy.linalg import expm

_X = np.array([[0, 1], [1, 0]]); _Y = np.array([[0, -1j], [1j, 0]]); _Z = np.diag([1, -1])
def two(A, B):
    """Kronecker product with qiskit ordering: A on qubit 0 (rightmost), B on qubit 1."""
    return np.kron(B, A)

def max_dev_up_to_phase(U, V):
    U = np.asarray(U); V = np.asarray(V)
    i = np.unravel_index(np.argmax(np.abs(V)), V.shape)
    return float(np.max(np.abs(U - (U[i] / V[i]) * V)))

theta = 0.731
for name, gate, gen, sign in [("RXYplus", RXYplus, two(_X, _Y) + two(_Y, _X), -1),
                              ("RXYminus", RXYminus, two(_X, _Y) - two(_Y, _X), +1),
                              ("RXXplus (challenge_utils)", RXXplus, two(_X, _X) + two(_Y, _Y), -1)]:
    dev = max_dev_up_to_phase(Operator(gate(theta)).data, expm(sign * 1j * theta / 2 * gen))
    print(f"{name:28s} vs exp({'+' if sign > 0 else '-'}i theta/2 G): max deviation {dev:.1e}  {'OK' if dev < 1e-9 else 'WRONG'}")
'''))

_add(code(r'''
# grade your answer:
ff.grade_ex0_2(RXYplus, RXYminus)
'''))

# ---- 0.3 vacuum -------------------------------------------------------------------
_add(md(r"""
### Exercise 0.3 — The SC-ADAPT-VQE vacuum (2 pts)

Start from the strong-coupling vacuum (every site empty: $X$ on every even qubit), then apply the two pre-trained layers
$e^{i\theta_1 \hat O^V_{mh}(1)}$ and $e^{i\theta_3 \hat O^V_{mh}(3)}$ with $\theta_1 = 0.30738$, $\theta_3 = -0.04059$ (Table II of the paper).
The circuit implementations are Fig. 4(a),(b) of arXiv:2308.04481 (we ignore (c)); use the *right-hand, simplified* form of (b):

<img src="images/Oh135sim.png" width="1000"/>

`vacuum_prep_rotate_OV_1` is given. For `vacuum_prep_rotate_OV_3` follow Fig. 4(b) layer by layer, left to right:
$R_+(-\pi/2)$ on all even pairs $(2k, 2k+1)$; $R_-(-\theta)$ on all odd pairs $(2k+1, 2k+2)$; $R_+(\pi/2)$ on even pairs;
$R_+(-\pi/2)$ on odd pairs; $R_-(\theta)$ on the **interior** even pairs only ($k = 1,\dots,L-2$ — the boundary pairs are absent in the figure);
$R_+(\pi/2)$ on odd pairs.

*Hint (level 1):* the grader fingerprints the state at $L = 6$ and $L = 8$; the boundary treatment of the $R_-(\theta)$ layer is the
classic mistake (it shifts $\langle\hat\chi\rangle$ by 0.0065 at sites 1 and $2L-2$).
"""))

_add(code(r'''
# PROMPT: Complete the function so that the output circuit prepares the strong-coupling vacuum where all sites are empty.
def prep_strong_coupling_vacuum(L: int) -> QuantumCircuit:
    """All sites empty: electrons (even sites) |1>, positrons (odd sites) |0>."""
    qc = QuantumCircuit(2 * L)
    # BEGIN ANSWER
    for k in range(L):
        qc.x(2 * k)
    # END ANSWER
    return qc


def vacuum_prep_rotate_OV_1(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O^V_mh(1)) to qc (Fig. 4a of arXiv:2308.04481). [given]"""
    for k in range(L):
        qc.append(RXYminus(theta), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYminus(-theta), [2 * k + 1, 2 * k + 2])
    return qc


# PROMPT: Fill in the function below, which takes a circuit qc and adds exp(i theta O^V_mh(3)) (Fig. 4b, right-hand form).
def vacuum_prep_rotate_OV_3(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O^V_mh(3)) to qc (Fig. 4b of arXiv:2308.04481, simplified form)."""
    # BEGIN ANSWER
    for k in range(L):
        qc.append(RXYplus(-np.pi / 2), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYminus(-theta), [2 * k + 1, 2 * k + 2])
    for k in range(L):
        qc.append(RXYplus(np.pi / 2), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYplus(-np.pi / 2), [2 * k + 1, 2 * k + 2])
    for k in range(1, L - 1):                      # interior even pairs only
        qc.append(RXYminus(theta), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYplus(np.pi / 2), [2 * k + 1, 2 * k + 2])
    # END ANSWER
    return qc


def prep_vacuum(L: int, vacuum_prep_theta_OV_1: float, vacuum_prep_theta_OV_3: float) -> QuantumCircuit:
    """Circuit preparing the 2-step SC-ADAPT-VQE vacuum on L spatial sites (2L qubits)."""
    qc = prep_strong_coupling_vacuum(L)
    # PROMPT: Apply the two rotations generating the vacuum state
    # BEGIN ANSWER
    qc = vacuum_prep_rotate_OV_1(qc, vacuum_prep_theta_OV_1, L)
    qc = vacuum_prep_rotate_OV_3(qc, vacuum_prep_theta_OV_3, L)
    # END ANSWER
    return qc


# The pre-trained variational parameters from the paper:
vacuum_prep_theta_OV_1 = 0.30738
vacuum_prep_theta_OV_3 = -0.04059

qc_vacuum_init = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
print(qc_vacuum_init.num_qubits, "qubits,", qc_vacuum_init.decompose().count_ops())
'''))

_add(code(r'''
# grade your answer (the grader calls prep_vacuum(L, 0.30738, -0.04059) at L = 6 and 8):
ff.grade_ex0_3(prep_vacuum)
'''))

# ---- 0.4 wavepacket ---------------------------------------------------------------
_add(md(r"""
### Exercise 0.4 — The wavepacket (2 pts)

The wavepacket is created on top of the vacuum with two more pre-trained layers generated by the wavepacket pool operators
(Eq. (12) of the paper), centred on the lattice:

$$
\hat{O}_{mh}(1,1) =\frac{1}{2} \left[ \hat{X}_{L-1}  \hat{Y}_{L} - \hat{Y}_{L-1} \hat{X}_{L}  \right], \qquad
\hat{O}_{mh}(2,2) =\frac{1}{2} \left[ \hat{X}_{L-2} \hat{Z}_{L-1}  \hat{Y}_{L} - \hat{Y}_{L-2} \hat{Z}_{L-1} \hat{X}_{L}
\ - \  \left (\hat{X}_{L-1} \hat{Z}_{L}  \hat{Y}_{L+1} - \hat{Y}_{L-1}  \hat{Z}_{L} \hat{X}_{L+1} \right ) \right],
$$

with $\theta_{11} = -1.6492$, $\theta_{22} = -0.3281$. Fig. 6 of the paper gives the circuits (use the simplified right-hand form of the bottom row):

<img src="images/2step_WP_ADAPT.png" width="1000"/>

*Hint (level 1):* $e^{i\theta\hat O_{mh}(2,2)}$ acts on the four central qubits $L-2, \dots, L+1$ and is four two-qubit gates:
$R_+(-\pi/2)$ on $(L-1, L)$, $R_+(-\theta)$ on $(L-2, L-1)$ and on $(L, L+1)$, $R_+(\pi/2)$ on $(L-1, L)$.
"""))

_add(code(r'''
def wave_prep_rotate_O_11(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O_mh(1,1)) on qubits L-1, L (Fig. 6, top). [given]"""
    qc.append(RXYminus(theta), [L - 1, L])
    return qc


# PROMPT: Complete the function that adds exp(i theta O_mh(2,2)) on the four central qubits L-2, L-1, L, L+1 (Fig. 6, bottom right)
def wave_prep_rotate_O_22(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O_mh(2,2)) on qubits L-2 .. L+1 (Fig. 6 bottom of arXiv:2401.08044, simplified form)."""
    # BEGIN ANSWER
    qc.append(RXYplus(-np.pi / 2), [L - 1, L])
    qc.append(RXYplus(-theta), [L - 2, L - 1])
    qc.append(RXYplus(-theta), [L, L + 1])
    qc.append(RXYplus(np.pi / 2), [L - 1, L])
    # END ANSWER
    return qc


def prep_wave(L: int, vacuum_prep_theta_OV_1: float, vacuum_prep_theta_OV_3: float,
              wave_prep_theta_O_11: float, wave_prep_theta_O_22: float) -> QuantumCircuit:
    """Circuit preparing the initial (centred) wavepacket state on L spatial sites."""
    qc = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    # PROMPT: Apply the two rotations that generate the wavepacket state from the vacuum state
    # BEGIN ANSWER
    qc = wave_prep_rotate_O_11(qc, wave_prep_theta_O_11, L)
    qc = wave_prep_rotate_O_22(qc, wave_prep_theta_O_22, L)
    # END ANSWER
    return qc


# The pre-trained variational parameters from the paper:
wave_prep_theta_O_11 = -1.6492
wave_prep_theta_O_22 = -0.3281

qc_wave_init = prep_wave(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
print(qc_wave_init.num_qubits, "qubits,", qc_wave_init.decompose().count_ops())
'''))

_add(code(r'''
# grade your answer (the grader calls prep_wave(L, th1, th3, th11, th22) at L = 6 and 8, with a random th22):
ff.grade_ex0_4(prep_wave)
'''))

_add(md(r"""
### The vacuum circuit for subtraction [no prompts]

The final quantity is $\mathcal{X}_j = \langle\hat\chi_j\rangle^{\rm wave} - \langle\hat\chi_j\rangle^{\rm vacuum}$. Hardware errors that hit
both circuits the same way cancel in the difference — but only if the two circuits have the same *structure*. We therefore apply the
wavepacket layers to the vacuum circuit as well, with an angle $\epsilon = 0.9\times10^{-4}$ that is tiny but non-zero, so that the
transpiler cannot remove them. (You will meet the same trick again — and quantify it — in Part 2.2.)
"""))

_add(code(r'''
def prep_vacuum_for_subtraction(L: int, eps: float = 0.9e-4) -> QuantumCircuit:
    """Vacuum circuit with the wavepacket layers applied at (almost) zero angle, so that its structure
    and noise match the wavepacket circuit (QDC convention)."""
    qc = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    qc = wave_prep_rotate_O_11(qc, eps, L)
    qc = wave_prep_rotate_O_22(qc, eps, L)
    return qc


qc_vacuum_init = prep_vacuum_for_subtraction(L)
print("wave / vacuum circuit 2q gate counts:",
      sum(1 for i in qc_wave_init.decompose().data if len(i.qubits) > 1),
      sum(1 for i in qc_vacuum_init.decompose().data if len(i.qubits) > 1))
'''))

_add(md(r"""
### Validation at $t=0$: exact condensates from an MPS simulation [no prompts]

The initial states are needed exactly for ODR (Part 3) and are a good first look at the signal. At $t = 0$ the circuits are shallow, so
Qiskit Aer's matrix-product-state simulator is exact to machine precision at $L = 34$. We compare with the QDC reference files: the wavepacket must
agree to $\sim 10^{-11}$; the vacuum differs by $\sim 10^{-4}$ because the QDC file was computed without the $\epsilon$ layers (that is the size of their effect).
"""))

_add(code(r'''
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

t0_ = time.time()
estimator_mps = AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state"}})
sim_results = estimator_mps.run([(qc_wave_init.decompose(reps=2), observables),
                                 (qc_vacuum_init.decompose(reps=2), observables)]).result()
chi_wave_exact = np.asarray(sim_results[0].data.evs, dtype=float)
chi_vacuum_exact = np.asarray(sim_results[1].data.evs, dtype=float)

chi_wave_exact_solution = np.loadtxt(f"{REF_DIR}/chi_wave_t0_sim_L34.txt")
chi_vacuum_exact_solution = np.loadtxt(f"{REF_DIR}/chi_vacuum_t0_sim_L34.txt")
print(f"MPS t=0 in {time.time() - t0_:.1f} s;  max |deviation from QDC files|: wave {np.max(np.abs(chi_wave_exact - chi_wave_exact_solution)):.1e},"
      f" vacuum {np.max(np.abs(chi_vacuum_exact - chi_vacuum_exact_solution)):.1e}")

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L), chi_wave_exact, "ro", label="Wavepacket (your circuit)")
ax.plot(range(2 * L), chi_vacuum_exact, "ko", label="Vacuum (your circuit)")
ax.plot(range(2 * L), chi_wave_exact_solution, "r--", linewidth=0.8, label="QDC reference")
ax.plot(range(2 * L), chi_vacuum_exact_solution, "k--", linewidth=0.8)
ax.set_ylim(-0.25, 1.9); ax.set_ylabel(r"$\langle\hat{\chi}_j\rangle$"); ax.set_xlabel(r"Fermion staggered site $j$"); ax.legend();
'''))

_add(code(r'''
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L), chi_wave_exact - chi_vacuum_exact, "ro", label="your circuits")
ax.plot(range(2 * L), chi_wave_exact_solution - chi_vacuum_exact_solution, "r--", linewidth=0.8, label="QDC reference")
ax.set_ylim(bottom=-0.25); ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel(r"Fermion staggered site $j$")
ax.set_title("Initial wavepacket, vacuum-subtracted (t = 0)"); ax.legend();
'''))

# ---- 0.5 Trotter -------------------------------------------------------------------
_add(md(r"""
### Exercise 0.5 — Second-order Trotter step and the four circuits (2 pts)

One second-order Suzuki–Trotter step of length $\delta t$ is (Fig. 8 of the paper)

$$
U_2(\delta t) = e^{-i\frac{\delta t}{2}\hat H_{kin}}\, e^{-i\delta t \hat H_{el}^{(1)}}\, e^{-i\delta t \hat H_m}\, e^{-i\frac{\delta t}{2}\hat H_{kin}},
$$

where the kinetic half-steps are themselves split into odd bonds $(1,2), (3,4), \dots$ and even bonds $(0,1), (2,3), \dots$. **The ordering is pinned** —
it is what the QDC reference files, the cached hardware run and the organizer MPS reference are computed with. Read Fig. 8 column by column:
kinetic $(\delta t/2)$ = *odd bonds first, then even bonds*; electric $(\delta t)$ = single-qubit $R_z$ layer then the four-qubit "barbell" $ZZ$ blocks;
mass $(\delta t)$; kinetic $(\delta t/2)$ = *even bonds first, then odd bonds*. Mirroring the sublattice order in the second half is what makes the step
symmetric (second order); it also lets the transpiler merge the two odd-bond layers at the junction of consecutive steps.

<img src="images/2ndTrott_circuits.png" width="1000"/>

The kinetic term $\tfrac{1}{2}(\hat\sigma^+_j\hat\sigma^-_{j+1} + {\rm h.c.}) = \tfrac{1}{4}(\hat X_j\hat X_{j+1} + \hat Y_j\hat Y_{j+1})$ on one bond is
implemented by $R^{(XX)}_{+}(\theta) = \exp[-i\tfrac{\theta}{2}(\hat X\hat X + \hat Y\hat Y)]$ (Fig. 5, right; imported from `challenge_utils`), so
a half step needs $\theta = \delta t/4$ on every bond:

<img src="images/xx_circ.png" width="700"/>

The barbell blocks (Appendix D of the paper) are given by `challenge_utils.trotter_step_electric_2q(qc, L, t, g)`. They do **not** contain
the single-qubit $R_z$ rotations of $\hat H_{el}^{(1)}$; that layer is written out for you below. The mass term is a layer of
$R_z\!\left((-1)^j m\,\delta t\right)$ — check the sign against $\hat H_m$ and Qiskit's $R_z(\phi) = e^{-i\phi Z/2}$.

`evolve_circuits` returns the *physics* circuit ($n$ steps forward) and the *calibration* circuit of the QDC recipe ($n/2$ steps forward,
$n/2$ steps backward, i.e. $\delta t \to -\delta t$), whose ideal output is the initial state. $n$ defaults to the QDC choice $2\lceil t/2\rceil$
($\delta t = 1$ for $t = 8$) and must be even.

*Hint (level 1):* odd bonds are `range(1, 2L-1, 2)`, even bonds `range(0, 2L-1, 2)`; the `time_step` argument may be negative. The grader's
fidelity check at $L = 6$ tells you if you swapped the sublattice order.
"""))

_add(code(r'''
# PROMPT: Implement a full second-order Trotter step as in Fig. 8 (pinned ordering, see the text above).
# The electric part (Rz layer + barbells) is already filled in.
def trotter_step(qc: QuantumCircuit, L: int, time_step: float, m: float, g: float) -> QuantumCircuit:
    """Second-order step: H_kin(t/2)[odd, even]  H_el(t)[Rz layer, barbells]  H_m(t)  H_kin(t/2)[even, odd]  (Fig. 8)."""
    n = 2 * L
    # BEGIN ANSWER
    # Implement H_kin over t/2 : odd bonds then even bonds, each R^{XX}_+(t/4)   (Fig. 8, first column)   # KEEP
    for j in range(1, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    for j in range(0, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    # Implement H_el over t: single-qubit Z part [ALREADY FILLED IN] + 4-qubit barbell ZZ part [given]   # KEEP
    for k in range(L // 2 - 1):   # KEEP
        qc.rz(g**2 * time_step, 2 * k)   # KEEP
        qc.rz(0.5 * g**2 * time_step, 2 * k + 1)   # KEEP
    qc.rz(0.5 * g**2 * time_step, L - 2)   # KEEP
    qc.rz(-0.5 * g**2 * time_step, L + 1)   # KEEP
    for k in range(1, L // 2):   # KEEP
        qc.rz(-0.5 * g**2 * time_step, L + 2 * k)   # KEEP
        qc.rz(-g**2 * time_step, L + 2 * k + 1)   # KEEP
    qc = trotter_step_electric_2q(qc, L, time_step, g)   # KEEP
    # Implement H_m over t   # KEEP
    for j in range(n):
        qc.rz((-1) ** j * m * time_step, j)
    # Finally, implement H_kin over t/2 in the mirrored sublattice order: even bonds then odd bonds (Fig. 8, last column)   # KEEP
    for j in range(0, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    for j in range(1, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    # END ANSWER
    return qc


def evolve_circuits(qc_init: QuantumCircuit, L: int, t: float, m: float, g: float, n_steps: int | None = None):
    """Physics circuit (n_steps forward) and QDC calibration circuit (n_steps/2 forward, n_steps/2 backward).

    n_steps defaults to the QDC choice 2*ceil(t/2) (dt = 1 for integer even t) and must be even.
    Returns (qc, qc_mitig)."""
    if n_steps is None:
        n_steps = int(2 * np.ceil(t / 2))
    assert n_steps % 2 == 0, "n_steps must be even for the forward/backward calibration circuit"
    time_step = t / n_steps
    qc = qc_init.copy()
    qc_mitig = qc_init.copy()
    # PROMPT: fill in the missing code: qc = n_steps forward steps, qc_mitig = n_steps/2 forward then n_steps/2 backward
    # BEGIN ANSWER
    for _ in range(n_steps):
        qc = trotter_step(qc, L, time_step, m, g)
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, time_step, m, g)
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, -time_step, m, g)
    # END ANSWER
    return qc, qc_mitig


# The four circuits for the hardware experiment (t = 8, 8 Trotter steps of dt = 1):
t = 8
qc_wave, qc_wave_mitig = evolve_circuits(qc_wave_init, L, t, m, g)
qc_vacuum, qc_vacuum_mitig = evolve_circuits(qc_vacuum_init, L, t, m, g)
circuits_all = [qc_wave, qc_wave_mitig, qc_vacuum, qc_vacuum_mitig]

def two_qubit_depth(qc: QuantumCircuit) -> int:
    return qc.depth(lambda i: (not getattr(i.operation, "_directive", False)) and len(i.qubits) > 1)

print("logical 2q depth (physics, calibration):", two_qubit_depth(qc_wave.decompose(reps=3)), two_qubit_depth(qc_wave_mitig.decompose(reps=3)))
'''))

_add(md(r"""
**Verify before you grade.** At $L = 6$ the calibration circuit must return to the initial state (fidelity 1 up to round-off), and one Trotter
step must approach $e^{-i\delta t \hat H^{(1)}}$ with a *third-order* local error: halving $\delta t$ divides the error by 8. You will build
$\hat H^{(1)}$ yourself in Part 1; here we only check the round trip.
"""))

_add(code(r'''
qc_init6 = prep_wave(6, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
qc6, qc6_mitig = evolve_circuits(qc_init6, 6, 4.0, m, g)
fid_return = abs(np.vdot(Statevector(qc_init6).data, Statevector(qc6_mitig).data)) ** 2
print(f"L=6, t=4: calibration circuit returns to the initial state with fidelity {fid_return:.12f}")
'''))

_add(code(r'''
# grade your answer (the grader runs trotter_step at L = 6 with random dt (ratio test + pinned-ordering fidelity) and
# evolve_circuits(prep_wave(6, ...), 6, 2, m, g)):
ff.grade_ex0_5(trotter_step, evolve_circuits, prep_wave)
'''))

_add(md(r"""
### The $t=8$ reference profile [no prompts]

The hardware run is scored against an organizer MPS simulation of *these* circuits (pinned Trotter ordering, $\delta t = 1$) at bond dimension 64
(`reference_data/mps_reference_L34.npz`, keys `chi_wave_t8_bd64`, `chi_vacuum_t8_bd64`; bond-128 copies are shipped for validation).
The QDC notebook shipped a bond-40 file; in Part 1.5 you will measure how converged each bond dimension is. If the organizer file is
missing the cell falls back to the QDC bond-40 file and says so.
"""))

_add(code(r'''
MPS_REF_PATH = f"{REF_DIR}/mps_reference_L34.npz"
if os.path.exists(MPS_REF_PATH):
    _ref = np.load(MPS_REF_PATH)
    chi_wave_ref_t8 = np.asarray(_ref["chi_wave_t8_bd64"], dtype=float)
    chi_vacuum_ref_t8 = np.asarray(_ref["chi_vacuum_t8_bd64"], dtype=float)
    REF_SOURCE = "organizer MPS, bond dimension 64"
else:
    warnings.warn("reference_data/mps_reference_L34.npz not found: falling back to the QDC bond-40 files")
    chi_wave_ref_t8 = np.loadtxt(f"{REF_DIR}/chi_wave_evolved_sim_L34_maxbond40.txt")
    chi_vacuum_ref_t8 = np.loadtxt(f"{REF_DIR}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
    REF_SOURCE = "QDC MPS file, bond dimension 40 (fallback)"
X_ref = chi_wave_ref_t8 - chi_vacuum_ref_t8

# the scoring window and the contrast used by the hardware metric (Part 4)
W = np.arange(25, 43)
PEAK_SITES, DIP_SITES = [31, 32, 35, 36], [33, 34]
C_ref = float(np.mean(X_ref[PEAK_SITES]) - np.mean(X_ref[DIP_SITES]))
print(f"reference: {REF_SOURCE};  peaks {X_ref[PEAK_SITES].round(3)}, dip {X_ref[DIP_SITES].round(3)}, contrast C_ref = {C_ref:.3f},"
      f" RMSE_0 = sqrt(mean X_ref[W]^2) = {np.sqrt(np.mean(X_ref[W] ** 2)):.3f}")

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L), X_ref, "g-o", label=REF_SOURCE)
ax.axvspan(W[0] - 0.5, W[-1] + 0.5, color="orange", alpha=0.15, label="scoring window W = 25..42")
ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel(r"Fermion staggered site $j$"); ax.set_title(f"L = {L}, t = 8: what the hardware should reproduce"); ax.legend();
'''))

# =====================================================================================
# Part 1 — physics you can verify
# =====================================================================================
_add(md(r"""
<span id="part1"></span>
# Part 1 — Physics you can verify (20 pts)

<div class="alert alert-block alert-success">

Before spending QPU time you should know, with numbers, how far the *ideal* circuit output is from the physics you claim to simulate.
Four approximations sit between $\hat H$ and the hardware profile: (i) the range-1 truncation of the electric interaction,
(ii) the 2-step SC-ADAPT-VQE initial states, (iii) the Trotter product formula with $\delta t = 1$, (iv) the finite bond dimension of the
classical reference. Each one gets a number in this part; together they form the error budget of Part 5. All exact calculations run at
$L \le 8$ (16 qubits, 65 536 amplitudes) with sparse matrices — every cell takes seconds.

</div>
"""))

# ---- 1.1 Hamiltonian ----------------------------------------------------------------
_add(md(r"""
### Exercise 1.1 — Build the Hamiltonian and check the charge sectors (5 pts)

Build `SparsePauliOp`s for $\hat H_m$, $\hat H_{kin}$, the full $\hat H_{el}$ and the truncated $\hat H_{el}^{(Q=0)}(1)$ (formula in the introduction;
it assumes even $L$), and `schwinger_hamiltonian(L, m, g, truncated=False)` $= \hat H_m + \hat H_{kin} + \hat H_{el}$.

* $\hat\sigma^+_j\hat\sigma^-_{j+1} + {\rm h.c.} = \tfrac{1}{2}(\hat X_j\hat X_{j+1} + \hat Y_j\hat Y_{j+1})$, so $\hat H_{kin} = \tfrac{1}{4}\sum_j (\hat X_j\hat X_{j+1} + \hat Y_j\hat Y_{j+1})$.
* $\hat H_{el} = \tfrac{g^2}{2}\sum_{j=0}^{2L-2}\big(\sum_{k\le j}\hat Q_k\big)^2$ with $\hat Q_k = -\tfrac{1}{2}[\hat Z_k + (-1)^k\hat I]$: build the cumulative charge
  as a `SparsePauliOp`, square it with `@`, and `simplify()`.
* **Charge-sector checks** (given below): the total charge $\hat Q = \sum_k \hat Q_k$ commutes with $\hat H$; the exact ground state has $\hat Q = 0$;
  the SC-ADAPT-VQE vacuum lives in the $Q = 0$ sector too (the $R^{(XY)}_\pm$ gates conserve charge). $\hat H_{el}^{(1)}$ is diagonal, so it trivially
  commutes with $\hat Q$ — but it is only meaningful inside the $Q = 0$ sector and differs from the full $\hat H_{el}$ by constants and by the dropped
  long-range terms, so **absolute energies of the two Hamiltonians are not comparable**; local dynamics is (Exercise 1.2).
* `E0_L8`: the ground-state energy of the *full* Hamiltonian at $L = 8$ with the $+\tfrac{m}{2}\hat I$ convention, from `scipy.sparse.linalg.eigsh`
  on `H.to_matrix(sparse=True)`.

*Hint (level 1):* the helper `pauli_op(n, {j: "Z", k: "Z"}, coeff)` below handles the little-endian bookkeeping; the grader compares Pauli
coefficient dictionaries at $L = 4, 6, 8$ (identity term excluded), so a single wrong boundary coefficient in the truncated formula is caught.
"""))

_add(code(r'''
from scipy.sparse.linalg import eigsh, expm_multiply

def pauli_op(n: int, ops: dict[int, str], coeff: float = 1.0) -> SparsePauliOp:
    """SparsePauliOp on n qubits with the Paulis in ops = {site: 'X'|'Y'|'Z'} (little-endian handled here)."""
    s = ["I"] * n
    for j, p in ops.items():
        s[n - 1 - j] = p
    return SparsePauliOp.from_list([("".join(s), coeff)])


# PROMPT: Build the four pieces of the Hamiltonian as SparsePauliOps on 2L qubits (conventions: see the introduction).
def mass_hamiltonian(L: int, m: float = 0.5) -> SparsePauliOp:
    """H_m = m/2 sum_j [(-1)^j Z_j + I]  (identity term included)."""
    n = 2 * L
    H = 0 * pauli_op(n, {})
    # BEGIN ANSWER
    for j in range(n):
        H += pauli_op(n, {j: "Z"}, m / 2 * (-1) ** j) + pauli_op(n, {}, m / 2)
    # END ANSWER
    return H.simplify()


def kinetic_hamiltonian(L: int) -> SparsePauliOp:
    """H_kin = 1/2 sum_j (sigma+_j sigma-_{j+1} + h.c.) = 1/4 sum_j (X_j X_{j+1} + Y_j Y_{j+1})."""
    n = 2 * L
    H = 0 * pauli_op(n, {})
    # BEGIN ANSWER
    for j in range(n - 1):
        H += pauli_op(n, {j: "X", j + 1: "X"}, 0.25) + pauli_op(n, {j: "Y", j + 1: "Y"}, 0.25)
    # END ANSWER
    return H.simplify()


def electric_hamiltonian_full(L: int, g: float = 0.3) -> SparsePauliOp:
    """H_el = g^2/2 sum_{j=0}^{2L-2} (sum_{k<=j} Q_k)^2 with Q_k = -1/2 (Z_k + (-1)^k I) (open boundaries)."""
    n = 2 * L
    H = 0 * pauli_op(n, {})   # accumulate the terms into H; the return statement below simplifies it  # KEEP
    # BEGIN ANSWER
    Q = [(pauli_op(n, {k: "Z"}, -0.5) + pauli_op(n, {}, -0.5 * (-1) ** k)) for k in range(n)]
    cum = 0 * pauli_op(n, {})
    for j in range(n - 1):
        cum = (cum + Q[j]).simplify()
        H += (g**2 / 2) * (cum @ cum)
    # END ANSWER
    return H.simplify()


def electric_hamiltonian_truncated(L: int, g: float = 0.3) -> SparsePauliOp:
    """H_el^{(Q=0)}(1): the range-1 truncated, charge-zero-sector electric Hamiltonian (requires even L)."""
    assert L % 2 == 0, "the truncated electric Hamiltonian formula assumes even L"
    n = 2 * L
    half = L // 2
    H = 0 * pauli_op(n, {})   # accumulate the terms into H; the return statement below simplifies it  # KEEP
    # BEGIN ANSWER
    T = []  # (dict site->'Z', coefficient) -- all Z-type
    for k in range(half):
        T.append(({2 * k: "Z", 2 * k + 1: "Z"}, half - 0.75 - k))
        T.append(({L + 2 * k: "Z", L + 2 * k + 1: "Z"}, k + 0.25))
    for k in range(1, half - 1):
        T.append(({2 * k: "Z"}, 1.0))
        T.append(({2 * k + 1: "Z"}, 0.5))
        T.append(({L + 2 * k: "Z"}, -0.5))
        T.append(({L + 2 * k + 1: "Z"}, -1.0))
    T += [({0: "Z"}, 1.0), ({1: "Z"}, 0.5), ({L - 2: "Z"}, 0.5),
          ({L + 1: "Z"}, -0.5), ({2 * L - 2: "Z"}, -0.5), ({2 * L - 1: "Z"}, -1.0)]
    for k in range(half - 1):
        c1, c2 = half - 1.25 - k, half - 1.75 - k
        T.append(({2 * k: "Z", 2 * k + 2: "Z"}, c1))
        T.append(({2 * k + 1: "Z", 2 * k + 2: "Z"}, c1))
        T.append(({2 * k: "Z", 2 * k + 3: "Z"}, c2))
        T.append(({2 * k + 1: "Z", 2 * k + 3: "Z"}, c2))
        c3, c4 = k + 0.25, k + 0.75
        T.append(({L + 2 * k + 2: "Z", L + 2 * k: "Z"}, c3))
        T.append(({L + 2 * k + 3: "Z", L + 2 * k: "Z"}, c3))
        T.append(({L + 2 * k + 2: "Z", L + 2 * k + 1: "Z"}, c4))
        T.append(({L + 2 * k + 3: "Z", L + 2 * k + 1: "Z"}, c4))
    for ops, c in T:
        H += pauli_op(n, ops, (g**2 / 2) * c)
    # END ANSWER
    return H.simplify()


def schwinger_hamiltonian(L: int, m: float = 0.5, g: float = 0.3, truncated: bool = False) -> SparsePauliOp:
    """H = H_m + H_kin + H_el  (truncated=True uses H_el^{(Q=0)}(1))."""
    H = 0 * pauli_op(2 * L, {})   # accumulate the three terms into H; the return statement below simplifies it  # KEEP
    # BEGIN ANSWER
    Hel = electric_hamiltonian_truncated(L, g) if truncated else electric_hamiltonian_full(L, g)
    H = mass_hamiltonian(L, m) + kinetic_hamiltonian(L) + Hel
    # END ANSWER
    return H.simplify()


H4 = schwinger_hamiltonian(4)
print("L=4 full H:", len(H4), "Pauli terms;  truncated H:", len(schwinger_hamiltonian(4, truncated=True)), "terms")
'''))

_add(code(r'''
# Charge-sector checks and exact ground-state energies [no prompts except E0_L8]
def total_charge(L: int) -> SparsePauliOp:
    n = 2 * L
    return sum((pauli_op(n, {k: "Z"}, -0.5) + pauli_op(n, {}, -0.5 * (-1) ** k)) for k in range(n)).simplify()

def commutator_norm(A: SparsePauliOp, B: SparsePauliOp) -> float:
    C = (A @ B - B @ A).simplify()
    return float(np.linalg.norm(C.coeffs)) if len(C.coeffs) else 0.0

E0, E0_trunc, ground_states = {}, {}, {}
for LL in (4, 6, 8):
    Hf = schwinger_hamiltonian(LL)
    Ht = schwinger_hamiltonian(LL, truncated=True)
    Q = total_charge(LL)
    w, v = eigsh(Hf.to_matrix(sparse=True), k=1, which="SA", tol=1e-12)
    wt, _ = eigsh(Ht.to_matrix(sparse=True), k=1, which="SA", tol=1e-12)
    E0[LL], E0_trunc[LL], ground_states[LL] = float(w[0]), float(wt[0]), v[:, 0]
    gs = Statevector(v[:, 0])
    vac = Statevector(prep_vacuum(LL, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3))
    print(f"L={LL}: ||[H,Q]|| = {commutator_norm(Hf, Q):.1e}, ||[H_trunc,Q]|| = {commutator_norm(Ht, Q):.1e};  "
          f"ground state <Q> = {np.real(gs.expectation_value(Q)):+.1e}, <Q^2> = {np.real(gs.expectation_value(Q @ Q)):.1e};  "
          f"ADAPT vacuum <Q^2> = {np.real(vac.expectation_value(Q @ Q)):.1e}")
    print(f"      E0(full) = {E0[LL]:.5f}   E0(truncated) = {E0_trunc[LL]:.5f}   (difference {E0_trunc[LL] - E0[LL]:+.3f}: constants + dropped long-range terms)")

# PROMPT: the exact ground-state energy of the full Hamiltonian at L = 8 (+m/2*I per site convention)
# PARTICIPANT: E0_L8 = # float: ground-state energy of the FULL H at L = 8, in the stated convention
E0_L8 = E0[8]  # SOL
print(f"\nE0_L8 = {E0_L8:.5f}")
'''))

_add(code(r'''
# grade your answer (Pauli dictionaries at L = 4, 6, 8, identity term excluded; E0_L8 to 1e-4):
ff.grade_ex1_1(schwinger_hamiltonian, electric_hamiltonian_truncated, E0_L8)
'''))

# ---- 1.2 electric layer + truncation shift -----------------------------------------------
_add(md(r"""
### Exercise 1.2 — The electric layer is exact; the truncation is not (4 pts)

**(a)** The electric part of one Trotter step — the $R_z$ layer plus the barbells — is *not* an approximation: all terms of $\hat H_{el}^{(1)}$ are diagonal
and commute, so the layer must equal $e^{-it\hat H_{el}^{(1)}}$ exactly (up to a global phase). Write `electric_layer(L, t, g)` returning that circuit alone
(same code as inside `trotter_step`) and verify it on three random states at $L = 4$ and $L = 6$ — the two branches of the barbell tiling in
`trotter_step_electric_2q` — using `scipy.sparse.linalg.expm_multiply` for $e^{-it\hat H}\lvert\psi\rangle$ (never build the dense $2^{2L}\times 2^{2L}$ exponential).

**(b)** The truncation *is* an approximation. `truncation_shift` is the largest change of the vacuum-subtracted condensate that it causes at $L = 8$, $t = 4$,
under *exact* evolution:

$$
\texttt{truncation\_shift} = \max_j \left| \mathcal{X}^{\rm full}_j(t) - \mathcal{X}^{\rm trunc}_j(t) \right|, \qquad
\mathcal{X}_j(t) = \langle \psi_{\rm wave}(t)|\hat\chi_j|\psi_{\rm wave}(t)\rangle - \langle \psi_{\rm vac}(t)|\hat\chi_j|\psi_{\rm vac}(t)\rangle,
$$

with $\lvert\psi_{\rm wave}(0)\rangle$ = `prep_wave(8, ...)`, $\lvert\psi_{\rm vac}(0)\rangle$ = `prep_vacuum(8, ...)` and $\lvert\psi(t)\rangle = e^{-it\hat H}\lvert\psi(0)\rangle$
for $\hat H$ = full and truncated. Expect a number of order $10^{-2}$.

*Hint (level 1):* `Statevector(psi).expectation_value(obs)` with the observables from `chiral_condensate_observables(8)`; `expm_multiply(-1j * t * H_sparse, psi)`.
"""))

_add(code(r'''
# PROMPT: electric_layer(L, t, g): the Rz layer + trotter_step_electric_2q on a fresh 2L-qubit circuit (nothing else)
def electric_layer(L: int, t: float, g: float = 0.3) -> QuantumCircuit:
    """Circuit implementing exp(-i t H_el^{(Q=0)}(1)) (single-qubit Rz layer + barbell blocks)."""
    qc = QuantumCircuit(2 * L)
    # BEGIN ANSWER
    for k in range(L // 2 - 1):
        qc.rz(g**2 * t, 2 * k)
        qc.rz(0.5 * g**2 * t, 2 * k + 1)
    qc.rz(0.5 * g**2 * t, L - 2)
    qc.rz(-0.5 * g**2 * t, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * t, L + 2 * k)
        qc.rz(-g**2 * t, L + 2 * k + 1)
    qc = trotter_step_electric_2q(qc, L, t, g)
    # END ANSWER
    return qc


# (a) identity check on 3 random states at L = 4 and L = 6
rng = np.random.default_rng(1)
elec_infidelity = {}
for LL in (4, 6):
    n = 2 * LL
    tt = 0.37
    layer = electric_layer(LL, tt, g)
    Hel = electric_hamiltonian_truncated(LL, g).to_matrix(sparse=True)
    worst = 0.0
    for _ in range(3):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi /= np.linalg.norm(psi)
        via_circuit = Statevector(psi).evolve(layer).data
        via_expm = expm_multiply(-1j * tt * Hel, psi)
        worst = max(worst, 1.0 - abs(np.vdot(via_circuit, via_expm)) ** 2)
    elec_infidelity[LL] = worst
    print(f"L={LL}: max infidelity between electric_layer and exp(-i t H_el^(1)) over 3 random states: {worst:.1e}")
'''))

_add(code(r'''
# (b) truncation shift at L = 8, t = 4 under exact evolution
def chi_from_state(psi, L: int) -> np.ndarray:
    """<chi_j> for all j from a statevector (numpy array or Statevector)."""
    sv = Statevector(psi)
    return np.array([np.real(sv.expectation_value(o)) for o in chiral_condensate_observables(L)])

L8 = 8
psi_wave_8 = Statevector(prep_wave(L8, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)).data
psi_vac_8 = Statevector(prep_vacuum(L8, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)).data
H8_full = schwinger_hamiltonian(L8).to_matrix(sparse=True)
H8_trunc = schwinger_hamiltonian(L8, truncated=True).to_matrix(sparse=True)

def exact_X(H_sparse, t: float, L: int = L8) -> np.ndarray:
    """Vacuum-subtracted condensate after exact evolution of prep_wave / prep_vacuum for time t."""
    return chi_from_state(expm_multiply(-1j * t * H_sparse, psi_wave_8), L) - chi_from_state(expm_multiply(-1j * t * H_sparse, psi_vac_8), L)

# PROMPT: compute X_full_t4, X_trunc_t4 (exact evolution to t = 4 with the full / truncated H) and truncation_shift = max_j |X_full - X_trunc|
# BEGIN ANSWER
X_full_t4 = exact_X(H8_full, 4.0)
X_trunc_t4 = exact_X(H8_trunc, 4.0)
truncation_shift = float(np.max(np.abs(X_full_t4 - X_trunc_t4)))
# END ANSWER
print(f"truncation_shift (L=8, t=4) = {truncation_shift:.4f}")

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L8), X_full_t4, "g-o", label="exact, full H")
ax.plot(range(2 * L8), X_trunc_t4, "b--s", label=r"exact, truncated $H^{(1)}$")
ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel("site j"); ax.set_title("L = 8, t = 4: what the range-1 truncation does"); ax.legend();
'''))

_add(code(r'''
# grade your answer (electric_layer is compared with expm at L = 4 and 6 at a random t; truncation_shift to 1e-3):
ff.grade_ex1_2(electric_layer, truncation_shift)
'''))

# ---- 1.3 VQE quality ----------------------------------------------------------------
_add(md(r"""
### Exercise 1.3 — How good is the 2-step SC-ADAPT-VQE vacuum? (3 pts)

Compare `prep_vacuum(L, 0.30738, -0.04059)` with the exact ground state of the **full** Hamiltonian (you already have it in `ground_states`).
Fill the dictionaries

* `vqe_fidelity[L]` $= |\langle \psi_0 | \psi_{\rm ADAPT}\rangle|^2$,
* `vqe_energy_gap[L]` $= \langle \psi_{\rm ADAPT}|\hat H|\psi_{\rm ADAPT}\rangle - E_0 \ (> 0)$,

for $L \in \{4, 6, 8\}$. The angles were optimised at $L = 56$ in the paper and are used unchanged here, so the fidelity drops slowly with $L$
(the infidelity per site is roughly constant) — that is the *state-preparation* line of your error budget. Also compute the largest change of
$\langle\hat\chi_j\rangle$ between the ADAPT vacuum and the exact ground state (`vqe_chi_error`) so that it can be compared with the other error sources.
"""))

_add(code(r'''
# PROMPT: fill vqe_fidelity = {4: .., 6: .., 8: ..} and vqe_energy_gap = {4: .., 6: .., 8: ..} (floats)
vqe_fidelity, vqe_energy_gap, vqe_chi_error = {}, {}, {}
for LL in (4, 6, 8):
    psi_adapt = Statevector(prep_vacuum(LL, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3))
    H_full = schwinger_hamiltonian(LL)
    # BEGIN ANSWER
    vqe_fidelity[LL] = float(abs(np.vdot(ground_states[LL], psi_adapt.data)) ** 2)
    vqe_energy_gap[LL] = float(np.real(psi_adapt.expectation_value(H_full)) - E0[LL])
    # END ANSWER
    vqe_chi_error[LL] = float(np.max(np.abs(chi_from_state(psi_adapt, LL) - chi_from_state(ground_states[LL], LL))))
    print(f"L={LL}: fidelity {vqe_fidelity[LL]:.4f}   (1-F)/(2L) = {(1 - vqe_fidelity[LL]) / (2 * LL):.2e}   "
          f"E_ADAPT - E0 = {vqe_energy_gap[LL]:.5f}   max_j |d<chi_j>| = {vqe_chi_error[LL]:.4f}")
'''))

_add(code(r'''
# grade your answer:
ff.grade_ex1_3(vqe_fidelity, vqe_energy_gap)
'''))

# ---- 1.4 Trotter error ---------------------------------------------------------------
_add(md(r"""
### Exercise 1.4 — The Trotter error at $\delta t = 1$ is the biggest approximation (5 pts)

The hardware circuits use $\delta t = 1$ ($8$ steps for $t = 8$) because every step costs ~600 CZ gates. How much physics does that cost? At $L = 8$ compare
the *Trotterised* vacuum-subtracted profile with the *exact* evolution under the same truncated Hamiltonian $\hat H^{(1)}$
(so that only the product-formula error is measured):

$$
\texttt{trotter\_table}[(t, \delta t)] = \max_j \left| \mathcal{X}^{\rm Trotter}_j(t; \delta t) - \mathcal{X}^{\rm exact,\,trunc}_j(t) \right|,
\qquad t \in \{2, 4\},\ \delta t \in \{1, 0.5, 0.25\},
$$

where both the wavepacket and the vacuum are evolved with `trotter_step` ($t/\delta t$ steps each) starting from `prep_wave(8, ...)` / `prep_vacuum(8, ...)`.
A second-order formula has a global error $\propto \delta t^2$, so successive ratios $\texttt{err}(\delta t)/\texttt{err}(\delta t/2)$ should approach 4.

**Richardson extrapolation.** Because the leading error is $c\,\delta t^2$, the combination $\mathcal{X}_R = \tfrac{4\,\mathcal{X}(\delta t/2) - \mathcal{X}(\delta t)}{3}$ cancels it.
`richardson_error` $= \max_j|\mathcal{X}_R - \mathcal{X}^{\rm exact,trunc}|$ at $t = 4$ from $\delta t = 0.5$ and $0.25$ — it should be far below both inputs
(bonus B1 applies the same idea to the 68-qubit MPS profiles).

**Dominant error.** Set `dominant_error` to the string `"trotter"`, `"truncation"` or `"state_prep"` — whichever of `trotter_table[(4, 1)]`, `truncation_shift`
and `vqe_chi_error[8]` is largest at $L = 8$, $t = 4$.

*Hint (level 1):* the exact profiles `exact_X(H8_trunc, t)` are already defined; a 16-step, 16-qubit Trotter circuit takes ~2 s in `Statevector`.
"""))

_add(code(r'''
def trotter_X(t: float, dt: float, L: int = L8) -> np.ndarray:
    """Vacuum-subtracted condensate from Trotterised evolution (t/dt steps) of prep_wave / prep_vacuum at lattice size L."""
    n_steps = int(round(t / dt))
    qw = prep_wave(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
    qv = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    for _ in range(n_steps):
        qw = trotter_step(qw, L, dt, m, g)
        qv = trotter_step(qv, L, dt, m, g)
    return chi_from_state(Statevector(qw), L) - chi_from_state(Statevector(qv), L)


# PROMPT: fill X_trotter[(t, dt)] (the 2L-site profiles) and trotter_table[(t, dt)] =
# max_j |X_trotter - X_exact_trunc| for t in (2, 4), dt in (1, 0.5, 0.25); then the Richardson
# profile `X_R` at t = 4 from dt = 0.5 and 0.25, `richardson_error` (its max deviation from the
# exact profile) and `dominant_error` (a string; the plot below uses X_R).
t0_ = time.time()
trotter_table, X_trotter = {}, {}
X_exact_trunc = {2: exact_X(H8_trunc, 2.0), 4: exact_X(H8_trunc, 4.0)}
# BEGIN ANSWER
for tt in (2, 4):
    for dt in (1.0, 0.5, 0.25):
        X_trotter[(tt, dt)] = trotter_X(tt, dt)
        trotter_table[(tt, dt)] = float(np.max(np.abs(X_trotter[(tt, dt)] - X_exact_trunc[tt])))
X_R = (4 * X_trotter[(4, 0.25)] - X_trotter[(4, 0.5)]) / 3
richardson_error = float(np.max(np.abs(X_R - X_exact_trunc[4])))
_budget = {"trotter": trotter_table[(4, 1.0)], "truncation": truncation_shift, "state_prep": vqe_chi_error[8]}
dominant_error = max(_budget, key=_budget.get)
# END ANSWER

print(f"({time.time() - t0_:.1f} s)\n  t    dt=1      dt=0.5    dt=0.25   ratio(1/0.5)  ratio(0.5/0.25)")
for tt in (2, 4):
    e1, e2, e3 = (trotter_table[(tt, dt)] for dt in (1.0, 0.5, 0.25))
    print(f"  {tt}   {e1:.4f}    {e2:.4f}    {e3:.4f}     {e1 / e2:.2f}          {e2 / e3:.2f}")
print(f"\nRichardson (t=4, dt=0.5 & 0.25): max error {richardson_error:.2e}  (vs {trotter_table[(4, 0.25)]:.2e} for dt=0.25 alone)")
print(f"L=8, t=4 error budget: trotter(dt=1) {trotter_table[(4, 1.0)]:.4f} | truncation {truncation_shift:.4f} | state_prep {vqe_chi_error[8]:.4f}"
      f"  ->  dominant_error = '{dominant_error}'")

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L8), X_exact_trunc[4], "k-", linewidth=3, label=r"exact ($H^{(1)}$)")
for dt, sty in ((1.0, "r--o"), (0.5, "b--s"), (0.25, "g--^")):
    ax.plot(range(2 * L8), X_trotter[(4, dt)], sty, linewidth=1, label=f"Trotter dt={dt}")
ax.plot(range(2 * L8), X_R, "m:", linewidth=2, label="Richardson (0.5, 0.25)")
ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel("site j"); ax.set_title("L = 8, t = 4"); ax.legend(ncol=3);
'''))

_add(code(r'''
# grade your answer (table entries to 5 %, ratios in [3, 5], Richardson below dt=0.25 error, dominant_error string):
ff.grade_ex1_4(trotter_table, richardson_error, dominant_error)
'''))

# ---- 1.5 MPS convergence ------------------------------------------------------------
_add(md(r"""
### Exercise 1.5 — Is the classical reference converged? MPS bond-dimension scan at $L = 34$, $t = 8$ (3 pts)

The hardware result is compared with a *classical* simulation, so the reference has its own error: the matrix-product-state truncation.
Run the $t = 8$ physics circuits (`qc_wave`, `qc_vacuum`) through Aer's MPS estimator at bond dimensions $\chi \in \{8, 12, 20, 40\}$, time each run,
and compare the vacuum-subtracted profile $\mathcal{X}(\chi)$ with the organizer bond-64 profile:

* `mps_err[chi]` $= \max_j |\mathcal{X}_j(\chi) - \mathcal{X}_j(64)|$, `mps_seconds[chi]` = wall time for the pair of circuits, `X_bond40` = your $\chi = 40$ profile.
* Set `RUN_BOND_64 = True` at the top if you want to reproduce the bond-64 reference yourself (~2 min on 8 cores); otherwise it is loaded from `mps_reference_L34.npz`.

Use `matrix_product_state_truncation_threshold = 1e-10` and `seed_simulator = 7` like the organizer run, and `.decompose(reps=3)` before the estimator.

**Why the QDC bond-20 file is unusable.** The public QDC repository also shipped `chi_*_evolved_sim_L34_maxbond20.txt` (not included here). Those files
contain *two rows* (`np.loadtxt` returns shape `(2, 68)`): row 0 is the $t = 0$ profile (identical to `chi_*_t0_sim_L34.txt`) and row 1 is an evolved snapshot
whose vacuum-subtracted profile is **not mirror-symmetric** (0.604 vs 0.582 at the inner peaks, 0.521 vs 0.492 at the outer ones, asymmetry 0.14) and deviates
from the bond-40 profile by up to 0.11, while a fresh bond-20 run (yours, below) is mirror-symmetric to $10^{-4}$ and differs from that file by more than 1 —
the file is not reproducible and the severe truncation broke the CP symmetry of the exact dynamics. Subtracting the two files naively silently broadcasts
into a $2 \times 68$ array. The single-row bond-40 files are converged much better; your scan quantifies by how much, and the Fall Fest scores against bond 64
(bond 128 agrees with it to $< 10^{-3}$ in $\mathcal{X}$, even though the individual condensates still move by $\sim 0.015$ — the vacuum background cancels in the difference).

*Hint (level 1):* `AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": chi, ...}, "run_options": {"seed_simulator": 7}})`.
"""))

_add(code(r'''
# PROMPT: run the bond-dimension scan and fill mps_err, mps_seconds (dicts keyed by bond dimension) and X_bond40
bonds = [8, 12, 20, 40] + ([64] if RUN_BOND_64 else [])
X_by_bond, mps_seconds, chi_by_bond = {}, {}, {}
qc_wave_d, qc_vacuum_d = qc_wave.decompose(reps=3), qc_vacuum.decompose(reps=3)
for chi in bonds:
    # BEGIN ANSWER
    est = AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state",
                                                      "matrix_product_state_max_bond_dimension": chi,
                                                      "matrix_product_state_truncation_threshold": 1e-10},
                                  "run_options": {"seed_simulator": 7}})
    t0_ = time.time()
    res = est.run([(qc_wave_d, observables), (qc_vacuum_d, observables)]).result()
    mps_seconds[chi] = time.time() - t0_
    chi_by_bond[chi] = (np.asarray(res[0].data.evs, dtype=float), np.asarray(res[1].data.evs, dtype=float))
    X_by_bond[chi] = chi_by_bond[chi][0] - chi_by_bond[chi][1]
    # END ANSWER
    print(f"bond {chi:3d}: {mps_seconds[chi]:6.1f} s   centre X = {np.round(X_by_bond[chi][31:37], 3)}")

if REF_SOURCE.startswith("organizer"):
    X_64 = X_ref
else:
    warnings.warn("bond-64 reference missing: errors are measured against the QDC bond-40 file instead")
    X_64 = X_by_bond.get(64, X_ref)

# PARTICIPANT: mps_err = # {bond: max_j |X_j(bond) - X_j(64)|}
mps_err = {chi: float(np.max(np.abs(X_by_bond[chi] - X_64))) for chi in bonds}  # SOL
# PARTICIPANT: X_bond40 = # your bond-40 vacuum-subtracted profile (68 floats)
X_bond40 = X_by_bond[40]  # SOL

X_qdc40 = np.loadtxt(f"{REF_DIR}/chi_wave_evolved_sim_L34_maxbond40.txt") - np.loadtxt(f"{REF_DIR}/chi_vacuum_evolved_sim_L34_maxbond40.txt")
print("\nbond   max|X(bond) - X(64)|   seconds")
for chi in bonds:
    print(f"{chi:4d}   {mps_err[chi]:.4f}               {mps_seconds[chi]:.1f}")
print(f"QDC bond-40 file vs bond 64: {np.max(np.abs(X_qdc40 - X_64)):.4f};  your bond 40 vs QDC bond-40 file: {np.max(np.abs(X_bond40 - X_qdc40)):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].plot(range(2 * L), X_64, "k-", linewidth=3, label="bond 64 (organizer)")
for chi, sty in zip(bonds, ("r:", "b--", "g-.", "m-", "c-")):
    ax[0].plot(range(2 * L), X_by_bond[chi], sty, linewidth=1, label=f"bond {chi}")
ax[0].set_xlim(22, 45); ax[0].set_ylabel(r"$\mathcal{X}_j$"); ax[0].set_xlabel("site j"); ax[0].legend(ncol=2)
ax[1].loglog(bonds, [mps_err[c] for c in bonds], "o-", label="max error vs bond 64")
ax[1].loglog(bonds, [mps_seconds[c] for c in bonds], "s--", label="wall time [s]")
ax[1].set_xlabel("bond dimension"); ax[1].legend();
'''))

_add(code(r'''
# grade your answer (mps_err must decrease with the bond dimension and match the organizer scan; X_bond40 within 3e-3 of the organizer bond 40):
ff.grade_ex1_5(mps_err, mps_seconds, X_bond40)
'''))

# =====================================================================================
# Part 2 — engineering the 68-qubit experiment
# =====================================================================================
_add(md(r"""
<span id="part2"></span>
# Part 2 — Engineering the 68-qubit experiment (15 pts)

<div class="alert alert-block alert-success">

The four circuits must run on the same 68 physical qubits with the same noise, on a device you chose with its calibration data, within a QPU budget
you can predict. This part builds that: a common layout (2.1), an honest gate count and a calibration circuit whose noise really matches the physics
circuit (2.2), a calibration-aware chain search (2.3), and the flight plan with the Estimator options you will submit with (2.4).
Everything runs offline against `FakeKingston` (the frozen calibration snapshot of `ibm_kingston` that ships with `qiskit-ibm-runtime`); switching to
a real backend is one cell.

</div>
"""))

# ---- 2.1 backend + transpile ---------------------------------------------------------
_add(md(r"""
### Exercise 2.1 — One layout for four circuits (3 pts)

**Backend.** By default we use `FakeKingston()` — a 156-qubit Heron r2 with a snapshot of real calibration data in `backend.target`, so every
transpiler decision below is the one you would get on hardware (with an older calibration). The Open Plan devices are `ibm_fez`, `ibm_marrakesh`
and `ibm_kingston`; the commented block shows how to switch. Use `use_fractional_gates=False`: the barbell blocks and $R^{(XX)}_+$ are synthesised
into CZ gates, and the fractional `rzz` variant is bonus B3.

**Layout.** Transpile the deepest circuit (`qc_wave`, $t = 8$) with the preset pass manager at optimization level 3 and `seed_transpiler=42`. The
routing stage finds a swap-free 68-qubit chain on the heavy-hex lattice (Sabre + `VF2PostLayout`, which already scores candidate placements with the
calibration errors). Then pin *all four* circuits to that layout with `initial_layout=` and `layout_method="trivial"` and map the observables with
`SparsePauliOp.apply_layout(qc_isa.layout)`.

*Hint (level 1):* `generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42, initial_layout=layout, layout_method="trivial")`;
the layout is `qc_isa.layout.initial_index_layout(filter_ancillas=True)`.
"""))

_add(code(r'''
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh

backend = FakeKingston()          # offline stand-in for ibm_kingston (frozen calibration snapshot)

# ---- To run on a real Open-Plan device, uncomment (needs a saved account; never paste a token into the notebook): ----
# from qiskit_ibm_runtime import QiskitRuntimeService
# service = QiskitRuntimeService()                       # or QiskitRuntimeService(name="<saved-account-name>")
# backend = service.backend("ibm_kingston", use_fractional_gates=False)   # ibm_fez / ibm_marrakesh / ibm_kingston
# --------------------------------------------------------------------------------------------------------------------
print(backend.name, backend.num_qubits, "qubits; basis:", sorted(backend.target.operation_names))
qubit_coordinates = challenge_utils.get_qubit_coordinates(backend)   # for the heavy-hex plots
'''))

_add(code(r'''
# PROMPT: transpile qc_wave with the preset pass manager (optimization level 3, seed 42) into qc_isa and extract its layout
# BEGIN ANSWER
pm_ref = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
t0_ = time.time()
qc_isa = pm_ref.run(qc_wave)
print(f"transpiled in {time.time() - t0_:.1f} s")
# END ANSWER
# PARTICIPANT: layout = # list of 68 physical qubits: initial layout of qc_isa (filter_ancillas=True)
layout = qc_isa.layout.initial_index_layout(filter_ancillas=True)  # SOL
qc_isa_init_layout = list(layout)          # QDC name for the same thing (used again in Part 4)
final_layout = qc_isa.layout.final_index_layout()
print("layout:", layout)
print("swap-free (initial == final layout):", list(layout) == list(final_layout))
print(f"2q depth = {two_qubit_depth(qc_isa)}, ops = {dict(qc_isa.count_ops())}")
challenge_utils.plot_qubit_chain(layout, backend, qubit_coordinates)
'''))

_add(code(r'''
# PROMPT: build a pass manager PINNED to this layout and call it `pm_pinned` (exercise 2.2 reuses it),
# then transpile all four circuits in circuits_all with it into `circuits_all_isa`
# (initial_layout + layout_method="trivial"; keep the order [wave, wave_mitig, vacuum, vacuum_mitig])
# and map the observables into `observables_isa` with apply_layout.
# BEGIN ANSWER
pm_pinned = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42,
                                         initial_layout=layout, layout_method="trivial")
circuits_all_isa = [pm_pinned.run(qc) for qc in circuits_all]
observables_isa = [obs.apply_layout(circuits_all_isa[0].layout) for obs in observables]
# END ANSWER

# sanity checks: same initial and final layout for all four, ISA-compliant, observables on 156 qubits
assert all(list(qc.layout.initial_index_layout(filter_ancillas=True)) == list(layout) for qc in circuits_all_isa)
assert all(list(qc.layout.final_index_layout()) == list(final_layout) for qc in circuits_all_isa)
assert all(o.num_qubits == backend.num_qubits for o in observables_isa)
print("2q depth per circuit:", [two_qubit_depth(qc) for qc in circuits_all_isa])
print("CZ count per circuit:", [qc.count_ops().get("cz", 0) for qc in circuits_all_isa])

with open("submission/circuits_isa.qpy", "wb") as f:
    qpy.dump(circuits_all_isa, f)
'''))

_add(code(r'''
# grade your answer (ISA compliance vs backend.target, one common layout, 2q-depth range, observables mapped to the layout):
ff.grade_ex2_1(circuits_all_isa, observables_isa, backend)
'''))

# ---- 2.2 gate accounting + matched calibration circuit -------------------------------
_add(md(r"""
### Exercise 2.2 — Gate accounting and a calibration circuit whose noise actually matches (5 pts)

**Accounting.** Count the two-qubit gates of the *logical* physics circuit (`qc_wave.decompose(reps=4)`, i.e. after expanding $R^{(XY)}$, $R^{(XX)}$
and barbell gates into CX gates) — `n2q_logical` — and the CZ gates of the transpiled one — `n_cz_physics`. The transpiler finds ~14 % fewer: at every
junction between consecutive Trotter steps the last odd-bond $R^{(XX)}_+(\delta t/4)$ layer of one step and the first odd-bond layer of the next are
merged into one $R^{(XX)}_+(\delta t/2)$ block (2 CZ instead of 4), and the even-bond $R^{(XX)}_+$ gates that sit next to the barbell blocks are absorbed
into their CX chains by the two-qubit block consolidation.

**The problem with the QDC calibration circuit.** ODR assumes the calibration circuit sees the *same* noise as the physics circuit. In the QDC circuit
(forward $n/2$ steps, then backward $n/2$ steps) the turning point is $\cdots R^{(XX)}_+(\tfrac{\delta t}{4})_{\rm even} R^{(XX)}_+(\tfrac{\delta t}{4})_{\rm odd}
\,\big|\, R^{(XX)}_+(-\tfrac{\delta t}{4})_{\rm odd} R^{(XX)}_+(-\tfrac{\delta t}{4})_{\rm even}\cdots$: the odd layers cancel exactly, then the even layers become adjacent
and cancel too, and the transpiler removes all of them — about 170 CZ gates (3.4 % of the circuit) that the physics circuit *does* execute. A `barrier()` at the
turning point stops the cancellation but also stops the legitimate merge, so the calibration circuit then has *more* CZ gates than
the physics circuit (+66). The cell below shows both numbers.

**Your task: `evolve_circuits_matched(qc_init, L, t, m, g)`** — same signature and return value `(qc, qc_mitig)` as `evolve_circuits` — whose calibration
circuit (i) still returns to the initial state (fidelity $> 1 - 10^{-6}$ at $L = 6$) and (ii) transpiles, on the pinned layout, to a CZ count of **at least 99 % of
the physics circuit's (no upper bound) with every per-edge count within $\pm 2$**. That is exactly what the grader checks, and the barrier variant above (+66 CZ) already satisfies it;
the construction below matches the distribution exactly and is the one to use on hardware. A clean construction: build forward steps, then backward steps, but insert at the turning point
one $R^{(XX)}_+(\epsilon)$ on every **odd** bond (the sublattice that touches the junction) with a tiny non-cancelling angle ($\epsilon = 10^{-4}$). The three
odd-bond blocks at the junction then merge into one non-identity block (2 CZ, exactly like a physics junction) and the even layers stay separated by it
(like in the physics circuit). The fidelity cost is $\sim L\,\epsilon^2/2 \approx 10^{-8}$. If you find a better construction, use it — the grader only checks (i) and (ii).

*Hint (level 1):* per-edge CZ counts: iterate over `isa.data`, keep `cz`, key by `tuple(sorted(isa.find_bit(q).index for q in inst.qubits))`.
"""))

_add(code(r'''
from collections import Counter

def count_2q(qc: QuantumCircuit, reps: int = 4) -> int:
    """Number of multi-qubit gates in the fully decomposed logical circuit (barriers excluded)."""
    return sum(1 for i in qc.decompose(reps=reps).data
               if len(i.qubits) > 1 and not getattr(i.operation, "_directive", False))

def cz_per_edge(isa: QuantumCircuit) -> Counter:
    """{(physical qubit a, physical qubit b): number of CZ gates} for a transpiled circuit."""
    c = Counter()
    for inst in isa.data:
        if inst.operation.name == "cz":
            c[tuple(sorted(isa.find_bit(q).index for q in inst.qubits))] += 1
    return c

# PROMPT: n2q_logical = 2q gates of the logical qc_wave (decompose reps=4); n_cz_physics = CZ gates of circuits_all_isa[0]
# PARTICIPANT: n2q_logical = # int
n2q_logical = count_2q(qc_wave)  # SOL
# PARTICIPANT: n_cz_physics = # int
n_cz_physics = int(circuits_all_isa[0].count_ops().get("cz", 0))  # SOL
n_cz_mitig_naive = int(circuits_all_isa[1].count_ops().get("cz", 0))
n_junction_merges = (int(2 * np.ceil(t / 2)) - 1) * (L - 1) * 2
print(f"logical 2q gates (physics): {n2q_logical}   ->  {n_cz_physics} CZ after transpilation "
      f"({n2q_logical - n_cz_physics} fewer: {n_junction_merges} from the merged odd-bond layers at {int(2 * np.ceil(t / 2)) - 1} junctions x {L - 1} bonds x 2 CZ,"
      f" the rest from R_XX gates absorbed into neighbouring barbell blocks)")
print(f"QDC calibration circuit: {n_cz_mitig_naive} CZ  ->  {n_cz_physics - n_cz_mitig_naive} CZ fewer than the physics circuit")

# barrier variant: stops the cancellation, but also the legitimate merge at the turning point
qc_mid = qc_wave_init.copy()
for _ in range(4):
    qc_mid = trotter_step(qc_mid, L, 1.0, m, g)
qc_mid.barrier()
for _ in range(4):
    qc_mid = trotter_step(qc_mid, L, -1.0, m, g)
n_cz_mitig_barrier = int(pm_pinned.run(qc_mid).count_ops().get("cz", 0))
print(f"barrier variant:         {n_cz_mitig_barrier} CZ  ->  {n_cz_mitig_barrier - n_cz_physics:+d} vs the physics circuit")
'''))

_add(code(r'''
# PROMPT: evolve_circuits_matched(qc_init, L, t, m, g) -> (qc, qc_mitig) with a noise-matched calibration circuit
def evolve_circuits_matched(qc_init: QuantumCircuit, L: int, t: float, m: float, g: float,
                            n_steps: int | None = None, eps: float = 1e-4):
    """Physics circuit and a calibration circuit that (i) returns to the initial state and (ii) transpiles to the
    same CZ count / per-edge distribution as the physics circuit: forward n/2 steps, one R^{XX}_+(eps) on every odd
    bond at the turning point (prevents the exact cancellation without changing the state), backward n/2 steps."""
    if n_steps is None:
        n_steps = int(2 * np.ceil(t / 2))
    assert n_steps % 2 == 0
    time_step = t / n_steps
    n = 2 * L
    qc = qc_init.copy()
    qc_mitig = qc_init.copy()
    # BEGIN ANSWER
    for _ in range(n_steps):
        qc = trotter_step(qc, L, time_step, m, g)
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, time_step, m, g)
    for j in range(1, n - 1, 2):                       # turning point: tiny non-cancelling kinetic layer on the odd bonds
        qc_mitig.append(RXXplus(eps), [j, j + 1])
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, -time_step, m, g)
    # END ANSWER
    return qc, qc_mitig


# verification at L = 6, t = 4: return fidelity and CZ accounting on a pinned FakeKingston layout
qc6_phys, qc6_matched = evolve_circuits_matched(qc_init6, 6, 4.0, m, g)
fid_matched = abs(np.vdot(Statevector(qc_init6).data, Statevector(qc6_matched).data)) ** 2
pm6 = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
isa6 = pm6.run(qc6_phys)
layout6 = isa6.layout.initial_index_layout(filter_ancillas=True)
pm6_pinned = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42, initial_layout=layout6, layout_method="trivial")
cz_phys6, cz_matched6 = cz_per_edge(pm6_pinned.run(qc6_phys)), cz_per_edge(pm6_pinned.run(qc6_matched))
cz_naive6 = cz_per_edge(pm6_pinned.run(evolve_circuits(qc_init6, 6, 4.0, m, g)[1]))
print(f"L=6, t=4: return fidelity of the matched calibration circuit = {fid_matched:.10f} (1-F = {1 - fid_matched:.1e})")
print(f"          CZ physics {sum(cz_phys6.values())} | matched {sum(cz_matched6.values())} | QDC naive {sum(cz_naive6.values())};"
      f"  per-edge distribution identical: {cz_phys6 == cz_matched6}")

# and at full scale (L = 34, t = 8) on the pinned layout of Exercise 2.1
_, qc_wave_mitig_matched = evolve_circuits_matched(qc_wave_init, L, t, m, g)
_, qc_vacuum_mitig_matched = evolve_circuits_matched(qc_vacuum_init, L, t, m, g)
isa_matched = pm_pinned.run(qc_wave_mitig_matched)
print(f"L=34, t=8: CZ physics {n_cz_physics} | matched calibration {isa_matched.count_ops().get('cz', 0)} | 2q depth {two_qubit_depth(circuits_all_isa[0])} vs {two_qubit_depth(isa_matched)};"
      f"  per-edge identical: {cz_per_edge(circuits_all_isa[0]) == cz_per_edge(isa_matched)}")
'''))

_add(md(r"""
From here on the **matched** calibration circuits replace the QDC ones in `circuits_all` (order unchanged: wave, wave_mitig, vacuum, vacuum_mitig).
"""))

_add(code(r'''
qc_wave_mitig, qc_vacuum_mitig = qc_wave_mitig_matched, qc_vacuum_mitig_matched
circuits_all = [qc_wave, qc_wave_mitig, qc_vacuum, qc_vacuum_mitig]
circuits_all_isa = [pm_pinned.run(qc) for qc in circuits_all]
observables_isa = [obs.apply_layout(circuits_all_isa[0].layout) for obs in observables]
assert all(list(qc.layout.initial_index_layout(filter_ancillas=True)) == list(layout) for qc in circuits_all_isa)
print("CZ count per circuit:", [qc.count_ops().get("cz", 0) for qc in circuits_all_isa])
'''))

_add(code(r'''
# grade your answer (counts vs the organizer's; evolve_circuits_matched is run at L = 6 and 8: return fidelity + CZ accounting on the given layout):
ff.grade_ex2_2(n2q_logical, n_cz_physics, evolve_circuits_matched, prep_wave, backend, layout)
'''))

# ---- 2.3 chain selection --------------------------------------------------------------
_add(md(r"""
### Exercise 2.3 — Calibration-aware chain selection (5 pts)

The transpiler's layout is already good (its VF2 post-layout scores candidate layouts by gate and readout errors) — on FakeKingston it is within a few per cent of the best chain — but you can still do a little better with the calibration data in `backend.target` and *your* gate counts, and, more importantly, a search you wrote runs on *any* device, excludes dead couplers/readouts and flagged qubits explicitly (`exclude=`, used after the canary in Part 4), and is what the grader calls on a backend you do not know in advance. Write
`select_chain(backend, n_qubits=68) -> list[int]` returning a simple path of 68 distinct physical qubits (consecutive entries are coupled) that minimises

$$
\mathrm{cost}(\text{chain}) = \sum_{\text{bonds }(a,b)} n_{\rm CZ}\, \big[-\ln(1-\epsilon_{\rm CZ}(a,b))\big]
+ \sum_{q \in \text{chain}} \Big( n_{\rm sx}\,\big[-\ln(1-\epsilon_{\rm sx}(q))\big] + \big[-\ln(1-\epsilon_{\rm ro}(q))\big] \Big),
$$

i.e. the expected number of errors on the circuit, with $n_{\rm CZ} \approx$ `n_cz_physics`$/(2L-1) \approx 74$ CZ per bond and
$n_{\rm sx} \approx 150$ per qubit from your gate accounting (the log form keeps a dead coupler finite but very expensive). Rules:

* read $\epsilon_{\rm CZ}$ from `backend.target["cz"][(a, b)].error`, $\epsilon_{\rm sx}$ from `target["sx"]`, $\epsilon_{\rm ro}$ from `target["measure"]`;
* **exclude** every edge with $\epsilon_{\rm CZ} \ge 0.5$ and every qubit with $\epsilon_{\rm ro} \ge 0.5$ (FakeKingston has 7 dead couplers -- 14 directed entries in `target["cz"]` -- and one dead readout, qubit 146);
* search: randomised depth-first search for a 68-path from random start qubits, neighbours ordered by edge cost plus noise, random restarts within a
  time budget (default 15 s; the grader calls your function once more with the same budget), keep the cheapest. Do not enumerate all paths (there are billions).

The cell then prints the cost of the transpiler's layout and of your chain, re-transpiles the four circuits onto your chain and predicts the per-site
signal retention $r_j \approx \prod (1-\epsilon)$ over the gates that touch qubit $j$ — an optimistic estimate that ignores error propagation through the light cone
(on ibm_kingston the measured ODR retention at $t = 8$ was 0.15–0.33).

*Hint (level 1):* `target["cz"]` contains both orientations of every edge; `backend.coupling_map.get_edges()` gives the graph; keep the recursion depth in mind (68 is fine).
"""))

_add(code(r'''
def target_error_tables(backend, dead: float = 0.5):
    """(cz, ro, sx) error dictionaries from backend.target; missing values count as 1.0 (dead)."""
    tg = backend.target
    def err(props):
        e = getattr(props, "error", None) if props is not None else None
        return 1.0 if e is None or (isinstance(e, float) and math.isnan(e)) else float(e)
    cz = {}
    for q, p in tg["cz"].items():
        key = tuple(sorted(q))
        cz[key] = max(cz.get(key, 0.0), err(p))
    ro = {q[0]: err(p) for q, p in tg["measure"].items()}
    sx = {q[0]: err(p) for q, p in tg["sx"].items()}
    return cz, ro, sx


def _nl(e: float) -> float:
    return -math.log(max(1e-12, 1.0 - min(e, 0.999)))


def chain_cost(chain, backend, n_cz_per_bond: float = 74.0, n_sx_per_qubit: float = 150.0) -> float:
    """Expected number of errors on the circuit (log form) for a given chain of physical qubits."""
    cz, ro, sx = target_error_tables(backend)
    c = sum(n_cz_per_bond * _nl(cz[tuple(sorted((a, b)))]) for a, b in zip(chain[:-1], chain[1:]))
    c += sum(_nl(ro[q]) + n_sx_per_qubit * _nl(sx[q]) for q in chain)
    return float(c)


# PROMPT: select_chain(backend, n_qubits=68) -> list of n_qubits physical qubits forming a path, minimising chain_cost
def select_chain(backend, n_qubits: int = 68, time_budget: float = 15.0, seed: int = 0,
                 n_cz_per_bond: float = 74.0, n_sx_per_qubit: float = 150.0, dead: float = 0.5, exclude=()) -> list[int]:
    """Randomised DFS with restarts over the coupling graph (dead edges/qubits and `exclude`d qubits removed);
    returns the cheapest path found within time_budget seconds."""
    cz, ro, sx = target_error_tables(backend)
    exclude = set(int(q) for q in exclude)
    # BEGIN ANSWER
    adj = {q: set() for q in range(backend.target.num_qubits)}
    for (a, b), e in cz.items():
        if e < dead and ro[a] < dead and ro[b] < dead and a not in exclude and b not in exclude:
            adj[a].add(b); adj[b].add(a)
    edge_w = {e: n_cz_per_bond * _nl(v) for e, v in cz.items()}
    node_w = {q: _nl(ro[q]) + n_sx_per_qubit * _nl(sx[q]) for q in ro}
    rng = np.random.default_rng(seed)
    starts = [q for q in adj if adj[q]]
    best, best_cost = None, float("inf")
    t_start = time.time()
    while time.time() - t_start < time_budget:
        start = int(rng.choice(starts))
        noise = rng.uniform(0.0, 1.0)             # restart-specific amount of randomness in the neighbour order
        path, used, found = [start], {start}, None

        def dfs():
            nonlocal found
            if found is not None or time.time() - t_start > time_budget:
                return
            if len(path) == n_qubits:
                found = list(path)
                return
            last = path[-1]
            cand = [q for q in adj[last] if q not in used]
            cand.sort(key=lambda q: edge_w[tuple(sorted((last, q)))] + node_w[q] + noise * rng.exponential(0.05))
            for q in cand:
                path.append(q); used.add(q)
                dfs()
                if found is not None:
                    return
                path.pop(); used.discard(q)

        dfs()
        if found is not None:
            c = chain_cost(found, backend, n_cz_per_bond, n_sx_per_qubit)
            if c < best_cost:
                best, best_cost = found, c
    # END ANSWER
    # `best` (set inside the answer block) must hold the cheapest valid chain found
    assert best is not None and len(best) == n_qubits and len(set(best)) == n_qubits, "no valid chain found"
    return [int(q) for q in best]


n_cz_per_bond = n_cz_physics / (2 * L - 1)
n_sx_per_qubit = circuits_all_isa[0].count_ops().get("sx", 0) / (2 * L)
t0_ = time.time()
chain = select_chain(backend, n_qubits=2 * L, time_budget=15.0, seed=0, n_cz_per_bond=n_cz_per_bond, n_sx_per_qubit=n_sx_per_qubit)
print(f"select_chain: {time.time() - t0_:.1f} s;  chain = {chain}")

# validity + cost comparison
_edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
assert all(tuple(sorted((a, b))) in _edges for a, b in zip(chain[:-1], chain[1:])) and len(set(chain)) == 2 * L
cost_transpiler = chain_cost(layout, backend, n_cz_per_bond, n_sx_per_qubit)
cost_chain = chain_cost(chain, backend, n_cz_per_bond, n_sx_per_qubit)
print(f"expected errors per circuit: transpiler layout {cost_transpiler:.2f}  |  selected chain {cost_chain:.2f}  "
      f"({100 * (1 - cost_chain / cost_transpiler):+.1f} % fewer);  exp(-cost): {math.exp(-cost_transpiler):.2e} vs {math.exp(-cost_chain):.2e}")
challenge_utils.plot_qubit_chain(chain, backend, qubit_coordinates)
'''))

_add(code(r'''
# re-transpile the four circuits onto the selected chain (site j -> chain[j]) and predict the per-site retention
pm_chain = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42,
                                        initial_layout=chain, layout_method="trivial")
circuits_all_isa = [pm_chain.run(qc) for qc in circuits_all]
layout = circuits_all_isa[0].layout.initial_index_layout(filter_ancillas=True)
assert list(layout) == list(chain) and all(list(qc.layout.final_index_layout()) == list(chain) for qc in circuits_all_isa)
observables_isa = [obs.apply_layout(circuits_all_isa[0].layout) for obs in observables]
print("CZ per circuit on the chain:", [qc.count_ops().get("cz", 0) for qc in circuits_all_isa],
      " 2q depth:", [two_qubit_depth(qc) for qc in circuits_all_isa])

cz_tab, ro_tab, sx_tab = target_error_tables(backend)
cz_edges = cz_per_edge(circuits_all_isa[0])
sx_counts = Counter(circuits_all_isa[0].find_bit(inst.qubits[0]).index for inst in circuits_all_isa[0].data if inst.operation.name == "sx")
retention = np.ones(2 * L)
for j, q in enumerate(chain):
    r = (1 - ro_tab[q]) * (1 - sx_tab[q]) ** sx_counts.get(q, 0)
    for (a, b), n_cz in cz_edges.items():
        if q in (a, b):
            r *= (1 - cz_tab[(a, b)]) ** n_cz
    retention[j] = r
print(f"predicted per-site retention prod(1-eps) (optimistic, no light-cone propagation): min {retention.min():.2f}, median {np.median(retention):.2f}, max {retention.max():.2f}")
fig, ax = plt.subplots(1, 1, figsize=(15, 3.5))
ax.bar(range(2 * L), retention, color="steelblue")
ax.set_xlabel("site j"); ax.set_ylabel("predicted retention"); ax.set_ylim(0, 1);

with open("submission/circuits_isa.qpy", "wb") as f:
    qpy.dump(circuits_all_isa, f)
json.dump({"backend": backend.name, "chain": [int(q) for q in chain], "cost_chain": cost_chain, "cost_transpiler_layout": cost_transpiler},
          open("submission/layout.json", "w"), indent=1)
'''))

_add(code(r'''
# grade your answer (select_chain(backend) is called by the grader: valid 68-path, no dead edges/qubits, cost vs the frozen calibration):
ff.grade_ex2_3(select_chain, backend)
'''))

# ---- 2.4 flight plan + estimator options ------------------------------------------------
_add(md(r"""
### Exercise 2.4 — Flight plan and Estimator options (2 pts)

**Usage model.** On the Open Plan you pay in *QPU usage seconds*; a 4-PUB Estimator job with $N_{\rm twirl}$ randomisations and $S$ shots per randomisation executes
$4 \times N_{\rm twirl} \times S$ circuits. Empirically (documented 0.35 ms per execution, scaled to the observed 6-minute QDC run of 480 x 400 x 4 executions)

$$
\text{usage} \approx 2\,\text{s} + 0.45\,\text{ms} \times \#\text{executions per job}.
$$

The Fall Fest plan per team is: **canary** at $t = 0$ (2 PUBs x 16 twirls x 128 shots $= 4096$ executions, $\approx 4$ s), **main** run at $t = 8$
(4 PUBs x 64 twirls x 256 shots $= 65\,536$, $\approx 32$ s), **improvement** run ($\le 60$ s), total well below the 180 s cap with room for one retry.

**Options.** `make_estimator_options(backend)` returns an `EstimatorOptions` with gate *and* measurement twirling (`num_randomizations`, `shots_per_randomization`),
dynamical decoupling with the `XpXm` sequence, `resilience_level = 0` (the ODR run must be raw: ODR needs the calibration and physics signals with the *same* noise, and the runtime's
TREX/ZNE would also cost usage — they are for the 4.3 improvement run, through the options function of 3.2), `max_execution_time = 180` and `default_shots = num_randomizations * shots_per_randomization`.

`flight_plan` is a plain dict with keys `backend`, `mode` (`"job"`), `cap_s`, `seconds_per_execution`, `overhead_s_per_job`, `optimization_level`, one sub-dict
per run (`canary`, `main`, `improvement`) with `t`, `n_pubs`, `num_randomizations`, `shots_per_randomization`, `executions`, `usage_s`, the total
`total_usage_s` (the hardware cells of Part 4 refuse to submit unless it is `<= cap_s`), and — for the **main** run, at top level, what the grader checks —
`chain` (the 68 physical qubits of your ISA circuits), `num_randomizations`, `shots_per_randomization`, `dd_sequence`, `predicted_usage_s` (main-run usage from
the model above) and `predicted_sigma_X` (the statistical uncertainty per site you expect on $\mathcal{X}_j$ after ODR: shot noise $1/\sqrt{\text{shots}}$,
amplified by $1/\text{retention}$ and by $\sqrt 2$ for the difference of two circuits — use the retention estimate of 2.3).
"""))

_add(code(r'''
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime.options import EstimatorOptions

SECONDS_PER_EXECUTION = 0.45e-3
OVERHEAD_S_PER_JOB = 2.0

def predict_usage(n_pubs: int, num_randomizations: int, shots_per_randomization: int) -> tuple[int, float]:
    """(executions, predicted usage in seconds) for one Estimator job."""
    executions = n_pubs * num_randomizations * shots_per_randomization
    return executions, OVERHEAD_S_PER_JOB + SECONDS_PER_EXECUTION * executions


# PROMPT: make_estimator_options(backend, num_randomizations=64, shots_per_randomization=256) -> EstimatorOptions (see the text)
def make_estimator_options(backend, num_randomizations: int = 64, shots_per_randomization: int = 256) -> EstimatorOptions:
    options = EstimatorOptions()
    # BEGIN ANSWER
    options.resilience_level = 0
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True
    options.twirling.num_randomizations = num_randomizations
    options.twirling.shots_per_randomization = shots_per_randomization
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XpXm"
    options.default_shots = num_randomizations * shots_per_randomization
    options.max_execution_time = 180
    # END ANSWER
    return options


# PROMPT: fill the flight_plan dict (keys listed in the text) using predict_usage
# BEGIN ANSWER
flight_plan = {"backend": backend.name, "mode": "job", "cap_s": 180,
               "seconds_per_execution": SECONDS_PER_EXECUTION, "overhead_s_per_job": OVERHEAD_S_PER_JOB}
for run, (tt, n_pubs, n_tw, shots) in {"canary": (0, 2, 16, 128), "main": (8, 4, 64, 256), "improvement": (8, 4, 32, 256)}.items():
    ex, usage = predict_usage(n_pubs, n_tw, shots)
    flight_plan[run] = {"t": tt, "n_pubs": n_pubs, "num_randomizations": n_tw, "shots_per_randomization": shots,
                        "executions": ex, "usage_s": round(usage, 1)}
flight_plan["improvement"]["budget_s"] = 60
flight_plan["total_usage_s"] = round(sum(flight_plan[r]["usage_s"] for r in ("canary", "main", "improvement")), 1)
flight_plan["optimization_level"] = 3
# main-run summary at top level (what the grader checks): chain, twirling parameters, DD sequence, predicted usage and
# the predicted per-site statistical uncertainty of X = chi_wave - chi_vacuum after ODR:
#   sigma_chi ~ 1/sqrt(shots) per circuit, amplified by 1/retention by the ODR rescaling, and sqrt(2) for the difference.
flight_plan["chain"] = [int(q) for q in chain]
flight_plan["num_randomizations"] = flight_plan["main"]["num_randomizations"]
flight_plan["shots_per_randomization"] = flight_plan["main"]["shots_per_randomization"]
flight_plan["dd_sequence"] = "XpXm"
flight_plan["predicted_usage_s"] = flight_plan["main"]["usage_s"]   # main run; total_usage_s is what the Part-4 cells check against cap_s
shots_main = flight_plan["num_randomizations"] * flight_plan["shots_per_randomization"]
flight_plan["predicted_sigma_X"] = round(float(np.sqrt(2.0) / np.sqrt(shots_main) / np.median(retention)), 4)
# END ANSWER

estimator_options = make_estimator_options(backend, flight_plan["main"]["num_randomizations"], flight_plan["main"]["shots_per_randomization"])
print(json.dumps(flight_plan, indent=1))
print(f"\npredicted total usage {flight_plan['total_usage_s']} s of the {flight_plan['cap_s']} s cap "
      f"(margin {flight_plan['cap_s'] - flight_plan['total_usage_s']:.0f} s = one retry of the main run)")
print("twirling:", estimator_options.twirling)
print("DD:", estimator_options.dynamical_decoupling, "| resilience_level:", estimator_options.resilience_level,
      "| max_execution_time:", estimator_options.max_execution_time, "| default_shots:", estimator_options.default_shots)
json.dump(flight_plan, open("submission/flight_plan.json", "w"), indent=1)
'''))

_add(md(r"""
This is how the main run will be submitted in Part 4 (`EstimatorV2(mode=backend, options=estimator_options)`, one PUB per circuit, all 68 observables).
With a fake backend the Estimator would run *locally* on Aer — impossible for 68 qubits — so the cell only builds the PUBs and never calls `run`.
"""))

_add(code(r'''
pubs = [(qc, obs_list) for qc, obs_list in zip(circuits_all_isa, [observables_isa] * 4)]
estimator = EstimatorV2(mode=backend, options=estimator_options)
print(f"{len(pubs)} PUBs x {len(observables_isa)} observables on {backend.name};",
      f"job would execute {flight_plan['main']['executions']} circuits (~{flight_plan['main']['usage_s']} s)")
print("Estimator options ready:", estimator.options.twirling.num_randomizations, "twirls x", estimator.options.twirling.shots_per_randomization, "shots")
'''))

_add(code(r'''
# grade your answer (option fields, usage arithmetic, 4 ISA PUB circuits on one layout):
ff.grade_ex2_4(flight_plan, estimator_options, circuits_all_isa)
'''))

_add(md(r"""
<div class="alert alert-block alert-info">

**Checkpoint.** You now have four ISA circuits on a calibration-selected 68-qubit chain (`submission/circuits_isa.qpy`), a flight plan under the 180 s cap,
and a quantified error budget: truncation ($\sim 0.01$), state preparation ($1-F \sim 0.007$, up to $0.04$ in $\langle\chi_j\rangle$), Trotter at $\delta t = 1$ ($\sim 0.1$ at $L = 8$; bonus B1 measures it at $L = 34$),
MPS reference ($< 10^{-3}$). Part 3 builds the mitigation that has to fight the hardware noise on top of all this.

</div>
"""))
