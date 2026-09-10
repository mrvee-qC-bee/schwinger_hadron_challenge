
===== CELL 0 [markdown] tags=[] =====
# Hadron Dynamics in the Schwinger Model — *Trust, but Verify*
### UofT Qiskit Fall Fest 2026 edition

Based on [*Quantum Simulations of Hadron Dynamics in the Schwinger Model using 112 Qubits* — arXiv:2401.08044](https://arxiv.org/abs/2401.08044)
and [*Scalable Circuits for Preparing Ground States on Digital Quantum Computers: The Schwinger Model Vacuum on 100 Qubits* — arXiv:2308.04481](https://arxiv.org/abs/2308.04481).

> **Attribution.** The physics narrative, the figures in `images/`, the helper module `challenge_utils.py` and the
> warm-up fill-ins of Part 0 are adapted from the IBM Quantum Developer Conference 2025 challenge
> *"Hadron dynamics in the Schwinger model"* (`qiskit-community/qdc-challenges-2025`, Apache-2.0; see
> `LICENSE-qdc-challenges-2025`). Everything from Part 1 onwards — the verification tasks, the engineering
> tasks, the hand-written error mitigation, the hardware protocol and the grader — is new material written for
> the Fall Fest and is graded on content that does **not** appear in the public QDC notebook.


===== CELL 1 [markdown] tags=[] =====
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


===== CELL 2 [markdown] tags=[] =====
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
  convention without it. With this convention the exact ground-state energy at $L=8$ is $E_0 = -2.50901$
  (it would be $-6.50901$ without the constant). Reported energies must follow the stated convention; the grader
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
* Error mitigation follows the paper: Pauli twirling, dynamical decoupling, *operator decoherence renormalization* (ODR) with a
  calibration circuit per physics circuit, post-selection on the calibration signal, CP (mirror) symmetrisation and vacuum subtraction.

### Summary of circuits needed

For the single snapshot $t=8$ we need four circuits, which must run on the **same qubits** with the **same noise**:
1. wavepacket physics circuit, 2. wavepacket calibration (mitigation) circuit, 3. vacuum physics circuit, 4. vacuum calibration circuit.


===== CELL 3 [markdown] tags=[] =====
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
* **Mitigation you wrote yourself (Part 3, 20 pts).** ODR, its uncertainty and its bias, Pauli twirling and charge post-selection are
  written by you and rehearsed on a reduced noise model before touching hardware. `challenge_utils.postselection_and_mitigation` is only a cross-check.
* **Hardware (Part 4, 25 pts).** Canary run at $t=0$, main run at $t=8$, improvement run — scored by a fixed metric against an
  organizer **bond-64 MPS reference** (`reference_data/mps_reference_L34.npz`, validated against bond 128), with a fixed scoring window
  and contrast. The grader recomputes your mitigated profile from your raw expectation values with your own `odr_mitigate`.
* **Pinned Trotter ordering.** Kinetic (odd bonds, then even bonds) - electric - mass - kinetic (even, then odd), exactly Fig. 8. The hardware
  reference is computed for this ordering; any other valid ordering passes the ratio test but fails the fidelity check.
* **Error budget + viva (Part 5, 10 pts).** One page, one derivation, eight minutes, questions.

</div>


===== CELL 4 [markdown] tags=[] =====
<div class="alert alert-block alert-success">

## Rules

1. **QPU budget: 180 s of usage per team, total, hard cap.** Every hardware job must be submitted from this notebook with the
   options of Part 2.4 (`max_execution_time = 180`); the flight plan (canary <= 30 s, main run ~32 s, improvement run <= 60 s)
   leaves a margin for one retry. Usage above 180 s costs 5 points; a job that could not return is scored on the
   organizer-released fallback dataset, capped at 75 %.
2. **Open Plan, job mode.** No sessions, no `max_execution_time > 180`, no `resilience_level > 0` on the main run (you mitigate yourself).
   Never put credentials in the notebook; `RUN_ON_HARDWARE = False` keeps every hardware cell inert.
3. **Submission = the `submission/` folder** written by the grader cells (`score.json`, `exN.npz`, `circuits_isa.qpy`,
   hardware results and job ids) plus this notebook, executed. Run `ff.summary()` at the end.
4. **Local grader.** `fallfest_grader.py` runs offline; it never raises on a wrong answer, prints a hint, and stores a compact copy
   of your inputs so the organizers can re-grade. Re-running a grader cell overwrites the previous score for that exercise.
5. **Viva.** Part 5 is an 8-minute pitch of your error budget plus questions from the judges. Every team member should be able to
   explain every graded number.
6. Work as a team; do not share code across teams. Anything reused from the public QDC notebook must be attributed (it is, in Part 0).

</div>


===== CELL 5 [markdown] tags=[] =====
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
| | 3.2 | `twirl_circuit`, `postselect_charge`, charge witness | 5 |
| | 3.3 | noisy rehearsal at $L=6$, $t=4$ with your own ODR | 8 |
| 4 Hardware | 4.1 | canary at $t=0$ | 3 |
| | 4.2 | main run at $t=8$ (leaderboard metric) | 18 |
| | 4.3 | improvement run with z-score | 4 |
| 5 Error budget + defence | 5 | error-budget table, ODR bias derivation, pitch, viva | 10 |
| Bonus (max +6) | B1–B3 | Richardson at $L=34$, ODR on damping vs depolarising, fractional-gate barbell | 6 |

Every exercise follows the same pattern: a short theory block, a code cell with `# PROMPT:` and `# BEGIN ANSWER` / `# END ANSWER`
markers (fill in between the markers), and a grader cell. Level-1 hints are inline; ask a mentor for level 2.


===== CELL 6 [markdown] tags=[] =====
## Environment check

Target stack: `qiskit ~= 2.5`, `qiskit-ibm-runtime ~= 0.49`, `qiskit-aer ~= 0.17` (see `requirements.txt`). The grader is a local
module in this folder; it never contacts the network.


===== CELL 7 [code] tags=[] =====
import fallfest_grader as ff

ff.check_env()


===== CELL 8 [code] tags=[] =====
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
RUN_BOND_64 = False       # Part 1.5: also run the bond-64 MPS yourself (~6 min); the organizer arrays are loaded otherwise


===== CELL 9 [markdown] tags=[] =====
### Model parameters [no prompts]

The variational angles used below were trained for $m = 0.5$, $g = 0.3$ — do not change them. $L$ is the number of *spatial*
sites; the staggered electrons and positrons occupy $2L$ sites, i.e. $2L$ qubits.


===== CELL 10 [code] tags=[] =====
# DO NOT MODIFY (the variational circuit parameters used in the rest of this challenge are specific to these model parameters)
m = 0.5   # electron/positron mass
g = 0.3   # coupling
L = 34    # number of spatial lattice sites, to be captured by 2 x L = 68 qubits


===== CELL 11 [markdown] tags=[] =====
<span id="part0"></span>
# Part 0 — Warm-up: the QDC circuits, graded at other lattice sizes (10 pts)

<div class="alert alert-block alert-success">

This part reproduces the circuit construction of the QDC notebook: observables, the two-qubit building blocks, the SC-ADAPT-VQE
vacuum and wavepacket circuits, and the second-order Trotter step. Every function is graded by calling it at $L \in \{4, 6, 8\}$
(and random angles), so write functions of `L`, never hard-code 68 qubits. By the end of Part 0 you will have the four $t=8$
circuits, exact $t=0$ condensates from an MPS simulation, and the $t=8$ reference profile.

</div>


===== CELL 12 [markdown] tags=[] =====
### Exercise 0.1 — Chiral condensate observables (2 pts)

$\hat{\chi}_j = (-1)^j \hat Z_j + \hat{I}$ for $j = 0, \dots, 2L-1$. Return a list of $2L$ `SparsePauliOp`s, one per site, each a weighted
sum of two Pauli strings on $2L$ qubits.

*Hint (level 1):* remember little-endian — site $j$ is the character at position $2L-1-j$ of the string, or use
`SparsePauliOp.from_sparse_list`. Check: `observables[1]` must contain `-1.0 * Z` on qubit 1.


===== CELL 13 [code] tags=[] =====
# PROMPT: Complete the function so that it returns the list [chi_0, chi_1, ..., chi_{2L-1}] of SparsePauliOps.
def chiral_condensate_observables(L: int) -> list[SparsePauliOp]:
    """chi_j = (-1)^j Z_j + I for j = 0 .. 2L-1 (list of 2L SparsePauliOps on 2L qubits)."""
    n = 2 * L
    obs = []
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return obs


observables = chiral_condensate_observables(L)
print(len(observables), "observables;  chi_0 =", observables[0], "\n                 chi_1 =", observables[1])


===== CELL 14 [code] tags=[] =====
# grade your answer:
ff.grade_ex0_1(chiral_condensate_observables)


===== CELL 15 [markdown] tags=[] =====
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


===== CELL 16 [code] tags=[] =====
# PROMPT: Complete the functions below so that they output a gate implementing R^{(XY)}_{\pm}(\theta) on two qubits (Fig. 5).
# NOTE: Do not add barriers to the qc because this will throw an error when converting the circuit to a gate object.
def RXYplus(theta: float) -> Gate:
    """R^{(XY)}_+(theta) = exp(-i theta/2 (XY + YX))."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    gate = qc.to_gate(label=rf"$R^{{XY}}_{{+}}({theta:.4g})$")
    return gate


def RXYminus(theta: float) -> Gate:
    """R^{(XY)}_-(theta) = exp(+i theta/2 (XY - YX)) = exp(i theta O) with O = (XY - YX)/2."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    gate = qc.to_gate(label=rf"$R^{{XY}}_{{-}}({theta:.4g})$")
    return gate


===== CELL 17 [markdown] tags=[] =====
**Verify before you grade.** The cell below compares your gates with the matrix exponentials of the generators (up to a global phase).
`Operator` on a 2-qubit gate is cheap; this is the pattern you will use throughout Part 1.


===== CELL 18 [code] tags=[] =====
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


===== CELL 19 [code] tags=[] =====
# grade your answer:
ff.grade_ex0_2(RXYplus, RXYminus)


===== CELL 20 [markdown] tags=[] =====
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


===== CELL 21 [code] tags=[] =====
# PROMPT: Complete the function so that the output circuit prepares the strong-coupling vacuum where all sites are empty.
def prep_strong_coupling_vacuum(L: int) -> QuantumCircuit:
    """All sites empty: electrons (even sites) |1>, positrons (odd sites) |0>."""
    qc = QuantumCircuit(2 * L)
    # BEGIN ANSWER
    # YOUR CODE HERE
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
    # YOUR CODE HERE
    # END ANSWER
    return qc


def prep_vacuum(L: int, vacuum_prep_theta_OV_1: float, vacuum_prep_theta_OV_3: float) -> QuantumCircuit:
    """Circuit preparing the 2-step SC-ADAPT-VQE vacuum on L spatial sites (2L qubits)."""
    qc = prep_strong_coupling_vacuum(L)
    # PROMPT: Apply the two rotations generating the vacuum state
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return qc


# The pre-trained variational parameters from the paper:
vacuum_prep_theta_OV_1 = 0.30738
vacuum_prep_theta_OV_3 = -0.04059

qc_vacuum_init = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
print(qc_vacuum_init.num_qubits, "qubits,", qc_vacuum_init.decompose().count_ops())


===== CELL 22 [code] tags=[] =====
# grade your answer (the grader calls prep_vacuum(L, 0.30738, -0.04059) at L = 6 and 8):
ff.grade_ex0_3(prep_vacuum)


===== CELL 23 [markdown] tags=[] =====
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


===== CELL 24 [code] tags=[] =====
def wave_prep_rotate_O_11(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O_mh(1,1)) on qubits L-1, L (Fig. 6, top). [given]"""
    qc.append(RXYminus(theta), [L - 1, L])
    return qc


# PROMPT: Complete the function that adds exp(i theta O_mh(2,2)) on the four central qubits L-2, L-1, L, L+1 (Fig. 6, bottom right)
def wave_prep_rotate_O_22(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Adds exp(i theta O_mh(2,2)) on qubits L-2 .. L+1 (Fig. 6 bottom of arXiv:2401.08044, simplified form)."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return qc


def prep_wave(L: int, vacuum_prep_theta_OV_1: float, vacuum_prep_theta_OV_3: float,
              wave_prep_theta_O_11: float, wave_prep_theta_O_22: float) -> QuantumCircuit:
    """Circuit preparing the initial (centred) wavepacket state on L spatial sites."""
    qc = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    # PROMPT: Apply the two rotations that generate the wavepacket state from the vacuum state
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return qc


# The pre-trained variational parameters from the paper:
wave_prep_theta_O_11 = -1.6492
wave_prep_theta_O_22 = -0.3281

qc_wave_init = prep_wave(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
print(qc_wave_init.num_qubits, "qubits,", qc_wave_init.decompose().count_ops())


===== CELL 25 [code] tags=[] =====
# grade your answer (the grader calls prep_wave(L, th1, th3, th11, th22) at L = 6 and 8, with a random th22):
ff.grade_ex0_4(prep_wave)


===== CELL 26 [markdown] tags=[] =====
### The vacuum circuit for subtraction [no prompts]

The final quantity is $\mathcal{X}_j = \langle\hat\chi_j\rangle^{\rm wave} - \langle\hat\chi_j\rangle^{\rm vacuum}$. Hardware errors that hit
both circuits the same way cancel in the difference — but only if the two circuits have the same *structure*. We therefore apply the
wavepacket layers to the vacuum circuit as well, with an angle $\epsilon = 0.9\times10^{-4}$ that is tiny but non-zero, so that the
transpiler cannot remove them. (You will meet the same trick again — and quantify it — in Part 2.2.)


===== CELL 27 [code] tags=[] =====
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


===== CELL 28 [markdown] tags=[] =====
### Validation at $t=0$: exact condensates from an MPS simulation [no prompts]

The initial states are needed exactly for ODR (Part 3) and are a good first look at the signal. At $t = 0$ the circuits are shallow, so
Qiskit Aer's matrix-product-state simulator is exact to machine precision at $L = 34$. We compare with the QDC reference files: the wavepacket must
agree to $\sim 10^{-11}$; the vacuum differs by $\sim 10^{-4}$ because the QDC file was computed without the $\epsilon$ layers (that is the size of their effect).


===== CELL 29 [code] tags=[] =====
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


===== CELL 30 [code] tags=[] =====
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L), chi_wave_exact - chi_vacuum_exact, "ro", label="your circuits")
ax.plot(range(2 * L), chi_wave_exact_solution - chi_vacuum_exact_solution, "r--", linewidth=0.8, label="QDC reference")
ax.set_ylim(bottom=-0.25); ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel(r"Fermion staggered site $j$")
ax.set_title("Initial wavepacket, vacuum-subtracted (t = 0)"); ax.legend();


===== CELL 31 [markdown] tags=[] =====
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


===== CELL 32 [code] tags=[] =====
# PROMPT: Implement a full second-order Trotter step as in Fig. 8 (pinned ordering, see the text above).
# The electric part (Rz layer + barbells) is already filled in.
def trotter_step(qc: QuantumCircuit, L: int, time_step: float, m: float, g: float) -> QuantumCircuit:
    """Second-order step: H_kin(t/2)[odd, even]  H_el(t)[Rz layer, barbells]  H_m(t)  H_kin(t/2)[even, odd]  (Fig. 8)."""
    n = 2 * L
    # BEGIN ANSWER
    # YOUR CODE HERE
    # Implement H_kin over t/2 : odd bonds then even bonds, each R^{XX}_+(t/4)   (Fig. 8, first column)
    # Implement H_el over t: single-qubit Z part [ALREADY FILLED IN] + 4-qubit barbell ZZ part [given]
    for k in range(L // 2 - 1):
        qc.rz(g**2 * time_step, 2 * k)
        qc.rz(0.5 * g**2 * time_step, 2 * k + 1)
    qc.rz(0.5 * g**2 * time_step, L - 2)
    qc.rz(-0.5 * g**2 * time_step, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * time_step, L + 2 * k)
        qc.rz(-g**2 * time_step, L + 2 * k + 1)
    qc = trotter_step_electric_2q(qc, L, time_step, g)
    # Implement H_m over t
    # Finally, implement H_kin over t/2 in the mirrored sublattice order: even bonds then odd bonds (Fig. 8, last column)
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
    # YOUR CODE HERE
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


===== CELL 33 [markdown] tags=[] =====
**Verify before you grade.** At $L = 6$ the calibration circuit must return to the initial state (fidelity 1 up to round-off), and one Trotter
step must approach $e^{-i\delta t \hat H^{(1)}}$ with a *third-order* local error: halving $\delta t$ divides the error by 8. You will build
$\hat H^{(1)}$ yourself in Part 1; here we only check the round trip.


===== CELL 34 [code] tags=[] =====
qc_init6 = prep_wave(6, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
qc6, qc6_mitig = evolve_circuits(qc_init6, 6, 4.0, m, g)
fid_return = abs(np.vdot(Statevector(qc_init6).data, Statevector(qc6_mitig).data)) ** 2
print(f"L=6, t=4: calibration circuit returns to the initial state with fidelity {fid_return:.12f}")


===== CELL 35 [code] tags=[] =====
# grade your answer (the grader runs trotter_step at L = 6 with random dt (ratio test + pinned-ordering fidelity) and
# evolve_circuits(prep_wave(6, ...), 6, 2, m, g)):
ff.grade_ex0_5(trotter_step, evolve_circuits, prep_wave)


===== CELL 36 [markdown] tags=[] =====
### The $t=8$ reference profile [no prompts]

The hardware run is scored against an organizer MPS simulation of *these* circuits (pinned Trotter ordering, $\delta t = 1$) at bond dimension 64
(`reference_data/mps_reference_L34.npz`, keys `chi_wave_t8_bd64`, `chi_vacuum_t8_bd64`; bond-128 copies are shipped for validation).
The QDC notebook shipped a bond-40 file; in Part 1.5 you will measure how converged each bond dimension is. If the organizer file is
missing the cell falls back to the QDC bond-40 file and says so.


===== CELL 37 [code] tags=[] =====
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


===== CELL 38 [markdown] tags=[] =====
<span id="part1"></span>
# Part 1 — Physics you can verify (20 pts)

<div class="alert alert-block alert-success">

Before spending QPU time you should know, with numbers, how far the *ideal* circuit output is from the physics you claim to simulate.
Four approximations sit between $\hat H$ and the hardware profile: (i) the range-1 truncation of the electric interaction,
(ii) the 2-step SC-ADAPT-VQE initial states, (iii) the Trotter product formula with $\delta t = 1$, (iv) the finite bond dimension of the
classical reference. Each one gets a number in this part; together they form the error budget of Part 5. All exact calculations run at
$L \le 8$ (16 qubits, 65 536 amplitudes) with sparse matrices — every cell takes seconds.

</div>


===== CELL 39 [markdown] tags=[] =====
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
* `E0_L8`: the ground-state energy of the *full* Hamiltonian at $L = 8$ with the $+\tfrac{m}{2}\hat I$ convention ($-2.50901$), from `scipy.sparse.linalg.eigsh`
  on `H.to_matrix(sparse=True)`.

*Hint (level 1):* the helper `pauli_op(n, {j: "Z", k: "Z"}, coeff)` below handles the little-endian bookkeeping; the grader compares Pauli
coefficient dictionaries at $L = 4, 6, 8$ (identity term excluded), so a single wrong boundary coefficient in the truncated formula is caught.


===== CELL 40 [code] tags=[] =====
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
    # YOUR CODE HERE
    # END ANSWER
    return H.simplify()


def kinetic_hamiltonian(L: int) -> SparsePauliOp:
    """H_kin = 1/2 sum_j (sigma+_j sigma-_{j+1} + h.c.) = 1/4 sum_j (X_j X_{j+1} + Y_j Y_{j+1})."""
    n = 2 * L
    H = 0 * pauli_op(n, {})
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return H.simplify()


def electric_hamiltonian_full(L: int, g: float = 0.3) -> SparsePauliOp:
    """H_el = g^2/2 sum_{j=0}^{2L-2} (sum_{k<=j} Q_k)^2 with Q_k = -1/2 (Z_k + (-1)^k I) (open boundaries)."""
    n = 2 * L
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return H.simplify()


def electric_hamiltonian_truncated(L: int, g: float = 0.3) -> SparsePauliOp:
    """H_el^{(Q=0)}(1): the range-1 truncated, charge-zero-sector electric Hamiltonian (requires even L)."""
    assert L % 2 == 0, "the truncated electric Hamiltonian formula assumes even L"
    n = 2 * L
    half = L // 2
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return H.simplify()


def schwinger_hamiltonian(L: int, m: float = 0.5, g: float = 0.3, truncated: bool = False) -> SparsePauliOp:
    """H = H_m + H_kin + H_el  (truncated=True uses H_el^{(Q=0)}(1))."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


H4 = schwinger_hamiltonian(4)
print("L=4 full H:", len(H4), "Pauli terms;  truncated H:", len(schwinger_hamiltonian(4, truncated=True)), "terms")


===== CELL 41 [code] tags=[] =====
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
E0_L8 = # float, expected -2.50901
print(f"\nE0_L8 = {E0_L8:.5f}")


===== CELL 42 [code] tags=[] =====
# grade your answer (Pauli dictionaries at L = 4, 6, 8, identity term excluded; E0_L8 to 1e-4):
ff.grade_ex1_1(schwinger_hamiltonian, electric_hamiltonian_truncated, E0_L8)


===== CELL 43 [markdown] tags=[] =====
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


===== CELL 44 [code] tags=[] =====
# PROMPT: electric_layer(L, t, g): the Rz layer + trotter_step_electric_2q on a fresh 2L-qubit circuit (nothing else)
def electric_layer(L: int, t: float, g: float = 0.3) -> QuantumCircuit:
    """Circuit implementing exp(-i t H_el^{(Q=0)}(1)) (single-qubit Rz layer + barbell blocks)."""
    qc = QuantumCircuit(2 * L)
    # BEGIN ANSWER
    # YOUR CODE HERE
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


===== CELL 45 [code] tags=[] =====
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
# YOUR CODE HERE
# END ANSWER
print(f"truncation_shift (L=8, t=4) = {truncation_shift:.4f}")

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L8), X_full_t4, "g-o", label="exact, full H")
ax.plot(range(2 * L8), X_trunc_t4, "b--s", label=r"exact, truncated $H^{(1)}$")
ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel("site j"); ax.set_title("L = 8, t = 4: what the range-1 truncation does"); ax.legend();


===== CELL 46 [code] tags=[] =====
# grade your answer (electric_layer is compared with expm at L = 4 and 6 at a random t; truncation_shift to 1e-3):
ff.grade_ex1_2(electric_layer, truncation_shift)


===== CELL 47 [markdown] tags=[] =====
### Exercise 1.3 — How good is the 2-step SC-ADAPT-VQE vacuum? (3 pts)

Compare `prep_vacuum(L, 0.30738, -0.04059)` with the exact ground state of the **full** Hamiltonian (you already have it in `ground_states`).
Fill the dictionaries

* `vqe_fidelity[L]` $= |\langle \psi_0 | \psi_{\rm ADAPT}\rangle|^2$,
* `vqe_energy_gap[L]` $= \langle \psi_{\rm ADAPT}|\hat H|\psi_{\rm ADAPT}\rangle - E_0 \ (> 0)$,

for $L \in \{4, 6, 8\}$. The angles were optimised at $L = 56$ in the paper and are used unchanged here, so the fidelity drops slowly with $L$
(the infidelity per site is roughly constant) — that is the *state-preparation* line of your error budget. Also compute the largest change of
$\langle\hat\chi_j\rangle$ between the ADAPT vacuum and the exact ground state (`vqe_chi_error`) so that it can be compared with the other error sources.


===== CELL 48 [code] tags=[] =====
# PROMPT: fill vqe_fidelity = {4: .., 6: .., 8: ..} and vqe_energy_gap = {4: .., 6: .., 8: ..} (floats)
vqe_fidelity, vqe_energy_gap, vqe_chi_error = {}, {}, {}
for LL in (4, 6, 8):
    psi_adapt = Statevector(prep_vacuum(LL, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3))
    H_full = schwinger_hamiltonian(LL)
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    vqe_chi_error[LL] = float(np.max(np.abs(chi_from_state(psi_adapt, LL) - chi_from_state(ground_states[LL], LL))))
    print(f"L={LL}: fidelity {vqe_fidelity[LL]:.4f}   (1-F)/(2L) = {(1 - vqe_fidelity[LL]) / (2 * LL):.2e}   "
          f"E_ADAPT - E0 = {vqe_energy_gap[LL]:.5f}   max_j |d<chi_j>| = {vqe_chi_error[LL]:.4f}")


===== CELL 49 [code] tags=[] =====
# grade your answer:
ff.grade_ex1_3(vqe_fidelity, vqe_energy_gap)


===== CELL 50 [markdown] tags=[] =====
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


===== CELL 51 [code] tags=[] =====
def trotter_X(t: float, dt: float, L: int = L8) -> np.ndarray:
    """Vacuum-subtracted condensate from Trotterised evolution (t/dt steps) of prep_wave / prep_vacuum at lattice size L."""
    n_steps = int(round(t / dt))
    qw = prep_wave(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
    qv = prep_vacuum(L, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3)
    for _ in range(n_steps):
        qw = trotter_step(qw, L, dt, m, g)
        qv = trotter_step(qv, L, dt, m, g)
    return chi_from_state(Statevector(qw), L) - chi_from_state(Statevector(qv), L)


# PROMPT: fill trotter_table = {(t, dt): max_j |X_trotter - X_exact_trunc|} for t in (2, 4), dt in (1, 0.5, 0.25);
# then richardson_error (t = 4, from dt = 0.5 and 0.25) and dominant_error (string).
t0_ = time.time()
trotter_table, X_trotter = {}, {}
X_exact_trunc = {2: exact_X(H8_trunc, 2.0), 4: exact_X(H8_trunc, 4.0)}
# BEGIN ANSWER
# YOUR CODE HERE
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


===== CELL 52 [code] tags=[] =====
# grade your answer (table entries to 5 %, ratios in [3, 5], Richardson below dt=0.25 error, dominant_error string):
ff.grade_ex1_4(trotter_table, richardson_error, dominant_error)


===== CELL 53 [markdown] tags=[] =====
### Exercise 1.5 — Is the classical reference converged? MPS bond-dimension scan at $L = 34$, $t = 8$ (3 pts)

The hardware result is compared with a *classical* simulation, so the reference has its own error: the matrix-product-state truncation.
Run the $t = 8$ physics circuits (`qc_wave`, `qc_vacuum`) through Aer's MPS estimator at bond dimensions $\chi \in \{8, 12, 20, 40\}$, time each run,
and compare the vacuum-subtracted profile $\mathcal{X}(\chi)$ with the organizer bond-64 profile:

* `mps_err[chi]` $= \max_j |\mathcal{X}_j(\chi) - \mathcal{X}_j(64)|$, `mps_seconds[chi]` = wall time for the pair of circuits, `X_bond40` = your $\chi = 40$ profile.
* Set `RUN_BOND_64 = True` at the top if you want to reproduce the bond-64 reference yourself (~6 min); otherwise it is loaded from `mps_reference_L34.npz`.

Use `matrix_product_state_truncation_threshold = 1e-10` and `seed_simulator = 7` like the organizer run, and `.decompose(reps=3)` before the estimator.

**Why the QDC bond-20 file is unusable.** The public QDC repository also shipped `chi_*_evolved_sim_L34_maxbond20.txt` (not included here). Those files
contain *two rows* (`np.loadtxt` returns shape `(2, 68)`): row 0 is the $t = 0$ profile (identical to `chi_*_t0_sim_L34.txt`) and row 1 is an evolved snapshot
whose vacuum-subtracted profile is **not mirror-symmetric** (0.604 vs 0.582 at the inner peaks, 0.521 vs 0.492 at the outer ones, asymmetry 0.14) and deviates
from the bond-40 profile by up to 0.11, while a fresh bond-20 run (yours, below) is mirror-symmetric to $10^{-4}$ and differs from that file by more than 1 —
the file is not reproducible and the severe truncation broke the CP symmetry of the exact dynamics. Subtracting the two files naively silently broadcasts
into a $2 \times 68$ array. The single-row bond-40 files are converged much better; your scan quantifies by how much, and the Fall Fest scores against bond 64
(bond 128 agrees with it to $< 10^{-3}$ in $\mathcal{X}$, even though the individual condensates still move by $\sim 0.015$ — the vacuum background cancels in the difference).

*Hint (level 1):* `AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": chi, ...}, "run_options": {"seed_simulator": 7}})`.


===== CELL 54 [code] tags=[] =====
# PROMPT: run the bond-dimension scan and fill mps_err, mps_seconds (dicts keyed by bond dimension) and X_bond40
bonds = [8, 12, 20, 40] + ([64] if RUN_BOND_64 else [])
X_by_bond, mps_seconds, chi_by_bond = {}, {}, {}
qc_wave_d, qc_vacuum_d = qc_wave.decompose(reps=3), qc_vacuum.decompose(reps=3)
for chi in bonds:
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    print(f"bond {chi:3d}: {mps_seconds[chi]:6.1f} s   centre X = {np.round(X_by_bond[chi][31:37], 3)}")

if REF_SOURCE.startswith("organizer"):
    X_64 = X_ref
else:
    warnings.warn("bond-64 reference missing: errors are measured against the QDC bond-40 file instead")
    X_64 = X_by_bond.get(64, X_ref)

mps_err = # {bond: max_j |X_j(bond) - X_j(64)|}
X_bond40 = # your bond-40 vacuum-subtracted profile (68 floats)

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


===== CELL 55 [code] tags=[] =====
# grade your answer (mps_err must decrease with the bond dimension and match the organizer scan; X_bond40 within 3e-3 of the organizer bond 40):
ff.grade_ex1_5(mps_err, mps_seconds, X_bond40)


===== CELL 56 [markdown] tags=[] =====
<span id="part2"></span>
# Part 2 — Engineering the 68-qubit experiment (15 pts)

<div class="alert alert-block alert-success">

The four circuits must run on the same 68 physical qubits with the same noise, on a device you chose with its calibration data, within a QPU budget
you can predict. This part builds that: a common layout (2.1), an honest gate count and a calibration circuit whose noise really matches the physics
circuit (2.2), a calibration-aware chain search (2.3), and the flight plan with the Estimator options you will submit with (2.4).
Everything runs offline against `FakeKingston` (the frozen calibration snapshot of `ibm_kingston` that ships with `qiskit-ibm-runtime`); switching to
a real backend is one cell.

</div>


===== CELL 57 [markdown] tags=[] =====
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


===== CELL 58 [code] tags=[] =====
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


===== CELL 59 [code] tags=[] =====
# PROMPT: transpile qc_wave with the preset pass manager (optimization level 3, seed 42) into qc_isa and extract its layout
# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER
layout = # list of 68 physical qubits: initial layout of qc_isa (filter_ancillas=True)
qc_isa_init_layout = list(layout)          # QDC name for the same thing (used again in Part 4)
final_layout = qc_isa.layout.final_index_layout()
print("layout:", layout)
print("swap-free (initial == final layout):", list(layout) == list(final_layout))
print(f"2q depth = {two_qubit_depth(qc_isa)}, ops = {dict(qc_isa.count_ops())}")
challenge_utils.plot_qubit_chain(layout, backend, qubit_coordinates)


===== CELL 60 [code] tags=[] =====
# PROMPT: transpile all four circuits in circuits_all onto exactly this layout (initial_layout + layout_method="trivial"),
# then map the observables with apply_layout. Keep the order [wave, wave_mitig, vacuum, vacuum_mitig].
# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER

# sanity checks: same initial and final layout for all four, ISA-compliant, observables on 156 qubits
assert all(list(qc.layout.initial_index_layout(filter_ancillas=True)) == list(layout) for qc in circuits_all_isa)
assert all(list(qc.layout.final_index_layout()) == list(final_layout) for qc in circuits_all_isa)
assert all(o.num_qubits == backend.num_qubits for o in observables_isa)
print("2q depth per circuit:", [two_qubit_depth(qc) for qc in circuits_all_isa])
print("CZ count per circuit:", [qc.count_ops().get("cz", 0) for qc in circuits_all_isa])

with open("submission/circuits_isa.qpy", "wb") as f:
    qpy.dump(circuits_all_isa, f)


===== CELL 61 [code] tags=[] =====
# grade your answer (ISA compliance vs backend.target, one common layout, 2q-depth range, observables mapped to the layout):
ff.grade_ex2_1(circuits_all_isa, observables_isa, backend)


===== CELL 62 [markdown] tags=[] =====
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
circuit (i) still returns to the initial state (fidelity $> 1 - 10^{-6}$ at $L = 6$) and (ii) transpiles to the **same CZ count and the same per-edge CZ
distribution** as the physics circuit on the pinned layout. A clean construction: build forward steps, then backward steps, but insert at the turning point
one $R^{(XX)}_+(\epsilon)$ on every **odd** bond (the sublattice that touches the junction) with a tiny non-cancelling angle ($\epsilon = 10^{-4}$). The three
odd-bond blocks at the junction then merge into one non-identity block (2 CZ, exactly like a physics junction) and the even layers stay separated by it
(like in the physics circuit). The fidelity cost is $\sim L\,\epsilon^2/2 \approx 10^{-8}$. If you find a better construction, use it — the grader only checks (i) and (ii).

*Hint (level 1):* per-edge CZ counts: iterate over `isa.data`, keep `cz`, key by `tuple(sorted(isa.find_bit(q).index for q in inst.qubits))`.


===== CELL 63 [code] tags=[] =====
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
n2q_logical = # int
n_cz_physics = # int
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


===== CELL 64 [code] tags=[] =====
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
    # YOUR CODE HERE
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


===== CELL 65 [markdown] tags=[] =====
From here on the **matched** calibration circuits replace the QDC ones in `circuits_all` (order unchanged: wave, wave_mitig, vacuum, vacuum_mitig).


===== CELL 66 [code] tags=[] =====
qc_wave_mitig, qc_vacuum_mitig = qc_wave_mitig_matched, qc_vacuum_mitig_matched
circuits_all = [qc_wave, qc_wave_mitig, qc_vacuum, qc_vacuum_mitig]
circuits_all_isa = [pm_pinned.run(qc) for qc in circuits_all]
observables_isa = [obs.apply_layout(circuits_all_isa[0].layout) for obs in observables]
assert all(list(qc.layout.initial_index_layout(filter_ancillas=True)) == list(layout) for qc in circuits_all_isa)
print("CZ count per circuit:", [qc.count_ops().get("cz", 0) for qc in circuits_all_isa])


===== CELL 67 [code] tags=[] =====
# grade your answer (counts vs the organizer's; evolve_circuits_matched is run at L = 6 and 8: return fidelity + CZ accounting on the given layout):
ff.grade_ex2_2(n2q_logical, n_cz_physics, evolve_circuits_matched, prep_wave, backend, layout)


===== CELL 68 [markdown] tags=[] =====
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
* **exclude** every edge with $\epsilon_{\rm CZ} \ge 0.5$ and every qubit with $\epsilon_{\rm ro} \ge 0.5$ (FakeKingston has 14 dead couplers and one dead readout);
* search: randomised depth-first search for a 68-path from random start qubits, neighbours ordered by edge cost plus noise, random restarts within a
  time budget (a few seconds), keep the cheapest. Do not enumerate all paths (there are billions).

The cell then prints the cost of the transpiler's layout and of your chain, re-transpiles the four circuits onto your chain and predicts the per-site
signal retention $r_j \approx \prod (1-\epsilon)$ over the gates that touch qubit $j$ — an optimistic estimate that ignores error propagation through the light cone
(on ibm_kingston the measured ODR retention at $t = 8$ was 0.15–0.33).

*Hint (level 1):* `target["cz"]` contains both orientations of every edge; `backend.coupling_map.get_edges()` gives the graph; keep the recursion depth in mind (68 is fine).


===== CELL 69 [code] tags=[] =====
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
    # YOUR CODE HERE
    # END ANSWER
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


===== CELL 70 [code] tags=[] =====
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


===== CELL 71 [code] tags=[] =====
# grade your answer (select_chain(backend) is called by the grader: valid 68-path, no dead edges/qubits, cost vs the frozen calibration):
ff.grade_ex2_3(select_chain, backend)


===== CELL 72 [markdown] tags=[] =====
### Exercise 2.4 — Flight plan and Estimator options (2 pts)

**Usage model.** On the Open Plan you pay in *QPU usage seconds*; a 4-PUB Estimator job with $N_{\rm twirl}$ randomisations and $S$ shots per randomisation executes
$4 \times N_{\rm twirl} \times S$ circuits. Empirically (documented 0.35 ms per execution, scaled to the observed 6-minute QDC run of 480 x 400 x 4 executions)

$$
\text{usage} \approx 2\,\text{s} + 0.45\,\text{ms} \times \#\text{executions per job}.
$$

The Fall Fest plan per team is: **canary** at $t = 0$ (2 PUBs x 16 twirls x 128 shots $= 4096$ executions, $\approx 4$ s), **main** run at $t = 8$
(4 PUBs x 64 twirls x 256 shots $= 65\,536$, $\approx 32$ s), **improvement** run ($\le 60$ s), total well below the 180 s cap with room for one retry.

**Options.** `make_estimator_options(backend)` returns an `EstimatorOptions` with gate *and* measurement twirling (`num_randomizations`, `shots_per_randomization`),
dynamical decoupling with the `XpXm` sequence, `resilience_level = 0` (you mitigate yourself in Part 3 — the runtime's TREX/ZNE would double the usage and hide
what you are doing), `max_execution_time = 180` and `default_shots = num_randomizations * shots_per_randomization`.

`flight_plan` is a plain dict with keys `backend`, `mode` (`"job"`), `cap_s`, `seconds_per_execution`, `overhead_s_per_job`, `optimization_level`, one sub-dict
per run (`canary`, `main`, `improvement`) with `t`, `n_pubs`, `num_randomizations`, `shots_per_randomization`, `executions`, `usage_s`, the total
`total_usage_s` (the hardware cells of Part 4 refuse to submit unless it is `<= cap_s`), and — for the **main** run, at top level, what the grader checks —
`chain` (the 68 physical qubits of your ISA circuits), `num_randomizations`, `shots_per_randomization`, `dd_sequence`, `predicted_usage_s` (main-run usage from
the model above) and `predicted_sigma_X` (the statistical uncertainty per site you expect on $\mathcal{X}_j$ after ODR: shot noise $1/\sqrt{\text{shots}}$,
amplified by $1/\text{retention}$ and by $\sqrt 2$ for the difference of two circuits — use the retention estimate of 2.3).


===== CELL 73 [code] tags=[] =====
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
    # YOUR CODE HERE
    # END ANSWER
    return options


# PROMPT: fill the flight_plan dict (keys listed in the text) using predict_usage
# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER

estimator_options = make_estimator_options(backend, flight_plan["main"]["num_randomizations"], flight_plan["main"]["shots_per_randomization"])
print(json.dumps(flight_plan, indent=1))
print(f"\npredicted total usage {flight_plan['total_usage_s']} s of the {flight_plan['cap_s']} s cap "
      f"(margin {flight_plan['cap_s'] - flight_plan['total_usage_s']:.0f} s = one retry of the main run)")
print("twirling:", estimator_options.twirling)
print("DD:", estimator_options.dynamical_decoupling, "| resilience_level:", estimator_options.resilience_level,
      "| max_execution_time:", estimator_options.max_execution_time, "| default_shots:", estimator_options.default_shots)
json.dump(flight_plan, open("submission/flight_plan.json", "w"), indent=1)


===== CELL 74 [markdown] tags=[] =====
This is how the main run will be submitted in Part 4 (`EstimatorV2(mode=backend, options=estimator_options)`, one PUB per circuit, all 68 observables).
With a fake backend the Estimator would run *locally* on Aer — impossible for 68 qubits — so the cell only builds the PUBs and never calls `run`.


===== CELL 75 [code] tags=[] =====
pubs = [(qc, obs_list) for qc, obs_list in zip(circuits_all_isa, [observables_isa] * 4)]
estimator = EstimatorV2(mode=backend, options=estimator_options)
print(f"{len(pubs)} PUBs x {len(observables_isa)} observables on {backend.name};",
      f"job would execute {flight_plan['main']['executions']} circuits (~{flight_plan['main']['usage_s']} s)")
print("Estimator options ready:", estimator.options.twirling.num_randomizations, "twirls x", estimator.options.twirling.shots_per_randomization, "shots")


===== CELL 76 [code] tags=[] =====
# grade your answer (option fields, usage arithmetic, 4 ISA PUB circuits on one layout):
ff.grade_ex2_4(flight_plan, estimator_options, circuits_all_isa)


===== CELL 77 [markdown] tags=[] =====
<div class="alert alert-block alert-info">

**Checkpoint.** You now have four ISA circuits on a calibration-selected 68-qubit chain (`submission/circuits_isa.qpy`), a flight plan under the 180 s cap,
and a quantified error budget: truncation ($\sim 0.01$), state preparation ($\sim 0.01$), Trotter at $\delta t = 1$ ($\sim 0.1$ at $L = 8$; bonus B1 measures it at $L = 34$),
MPS reference ($< 10^{-3}$). Part 3 builds the mitigation that has to fight the hardware noise on top of all this.

</div>


===== CELL 78 [markdown] tags=[] =====
<span id="part3"></span>
# Part 3 — Mitigation you wrote yourself (20 pts)

The public QDC notebook hands you `challenge_utils.postselection_and_mitigation` and lets the
runtime do the twirling. In this part you write the two mitigation ingredients yourself so that
you can (a) explain every line at the viva, (b) attach an *uncertainty* to the mitigated profile
and (c) rehearse the whole pipeline on a noisy simulator **before** spending your 180 s of QPU time.

| ex | what you build | pts |
|---|---|---|
| 3.1 | `odr_mitigate`, `odr_uncertainty`, `odr_bias` | 7 |
| 3.2 | `twirl_circuit`, `postselect_charge`, the charge witness | 5 |
| 3.3 | `reduced_noise_model` + full noisy rehearsal at $L=6$, $t=4$ | 8 |


===== CELL 79 [code] tags=[] =====
# Part 3 set-up: imports and three small helpers used in Parts 3-5 (no prompts)
import os, json, time, copy, warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp, Pauli
from qiskit.transpiler import generate_preset_pass_manager
import challenge_utils as cu
import fallfest_grader as ff

os.makedirs("submission", exist_ok=True)

# SC-ADAPT-VQE angles (Part 0) and the near-zero wavepacket angle of the noise-matched vacuum circuit
TH_OV1, TH_OV3, TH_O11, TH_O22 = 0.30738, -0.04059, -1.6492, -0.3281
EPS_VAC = 0.9e-4
T_FINAL = t          # t = 8 from Part 0; never reassign `t` below


def exact_chi(qc, L_):
    """Exact <chi_j> of a small circuit from its statevector (fine up to ~14 qubits)."""
    sv = Statevector(qc)
    return np.array([sv.expectation_value(o).real for o in chiral_condensate_observables(L_)])


def rmse(a, b):
    """Root-mean-square deviation between two profiles, ignoring NaN entries."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def cp_average(chi):
    """CP symmetry of the lattice model == mirror symmetry of the chain: average chi_j with chi_{2L-1-j}."""
    chi = np.asarray(chi, float)
    return 0.5 * (chi + chi[::-1])


===== CELL 80 [markdown] tags=[] =====
## 3.1 Operator decoherence renormalization, derived (7 pts)

**Pauli noise rescales Pauli expectation values.** After Pauli twirling (Part 2 options, or your own
frames in 3.2) the effective noise channel of a circuit is a *Pauli channel*, whose Pauli transfer
matrix is diagonal. A diagonal PTM can only shrink a Pauli expectation value towards zero:

$$\langle Z_j\rangle_{\rm meas} \;=\; f_j\,\langle Z_j\rangle_{\rm true}, \qquad 0 < f_j \le 1 .$$

**Calibration circuit.** The forward/backward circuit of Part 0 (``qc_*_mitig``: $n/2$ Trotter steps
with $+dt$, then $n/2$ with $-dt$) has the *same* gate content as the physics circuit but a known
ideal answer: it returns to the $t=0$ state, whose $\langle Z_j\rangle$ you computed exactly in
Part 1 (``chi_wave_exact``, ``chi_vacuum_exact``). Hence its measured values give the factor directly:

$$ f^{\rm cal}_j \;=\; \frac{\langle Z_j\rangle^{\rm cal}_{\rm meas}}{\langle Z_j\rangle^{\rm cal}_{\rm exact}}
\;=\; \frac{1-\chi^{\rm cal}_{j,\rm meas}}{1-\chi^{\rm cal}_{j,\rm exact}}, \qquad\text{because}\quad
1-\chi_j = -(-1)^j Z_j .$$

The mitigated condensate of the physics circuit is then

$$ \chi^{\rm mit}_j \;=\; 1 - \frac{1-\chi_{j,\rm meas}}{f^{\rm cal}_j} .$$

**Bias.** The physics circuit is *not* the calibration circuit: its noise factor is some
$f^{\rm phys}_j \neq f^{\rm cal}_j$ (different angles, different cancellations at the turning
point — Part 2.2). Inserting $\chi_{j,\rm meas} = 1 - f^{\rm phys}_j (1-\chi_{j,\rm true})$ gives

$$ \boxed{\;\delta\chi_j \;\equiv\; \chi^{\rm mit}_j - \chi_{j,\rm true} \;=\; (1-\chi_{j,\rm true})\Bigl(1-\frac{f^{\rm phys}_j}{f^{\rm cal}_j}\Bigr)\;} $$

A 10 % mismatch of the factors therefore shifts $\chi_j$ by 10 % of $(1-\chi_j)$ — *not* by 10 % of
the vacuum-subtracted signal $\mathcal X_j$, which is why the subtraction of Part 4 removes most of it
(the wave and vacuum circuits share the mismatch).

**Post-selection and mirror pooling.** Dividing by a tiny $f$ amplifies shot noise by $1/f$ and a
statistical fluctuation can even flip its sign. Sites whose factor is below a threshold (QDC uses
$0.01$) are dropped. Because the model is CP symmetric ($j \leftrightarrow 2L-1-j$) the two mirror
sites carry the same physics, so their *surviving* measured values and factors are averaged before the
division; if neither survives the site is reported as `NaN` (the grader penalises NaNs in the window,
so use the threshold sparingly).

**Why non-Pauli noise breaks ODR.** Amplitude damping maps $Z \to (1-\gamma) Z + \gamma\,I$: the
measured value acquires a *state-independent offset*, $\langle Z\rangle_{\rm meas} = f\langle
Z\rangle_{\rm true} + b$, so the ratio measured on the calibration state does not transfer to the
physics state. Coherent errors mix Paulis ($Z \to \cos\theta\,Z + \sin\theta\,Y$) with a
state-dependent effect. Twirling turns both into Pauli channels (the offset $b$ averages to zero over
the random $X$ frames — exactly in the limit of many frames) — that is what 3.2 demonstrates with the charge witness.

Write the three functions below (**do not call `challenge_utils.postselection_and_mitigation`** —
you may reproduce its logic, but you will be asked to explain your own code at the viva).


===== CELL 81 [code] tags=[] =====
# PROMPT: implement ODR yourself.
#   odr_mitigate(chi, chi_cal, chi_exact, L, suppression_threshold=0.01) -> (2L,) array, NaN where no data
#       chi       : measured chi_j of the physics circuit           (2L,)  [may also be (n, 2L)]
#       chi_cal   : measured chi_j of the forward/backward circuit   (2L,)
#       chi_exact : exact chi_j of the calibration circuit (= the t=0 state)
#     f_j = (1 - chi_cal_j)/(1 - chi_exact_j); pool site q with its mirror 2L-1-q: keep the members with
#     f > threshold, average their chi and their f, chi_mit = 1 - (1 - mean chi)/(mean f); mirror-symmetric output.
#   odr_uncertainty(chi, chi_std, chi_cal, chi_cal_std, chi_exact, L, ...) -> 1-sigma array (2L,) obtained by
#     resampling chi and chi_cal from Gaussians with the given stds (n_samples draws, fixed seed) and taking the
#     standard deviation of the mitigated values (NaN where the central estimate is NaN).
#   odr_bias(chi_true, f_phys, f_cal) -> (1 - chi_true) * (1 - f_phys/f_cal)   (broadcasts over arrays)

def odr_mitigate(chi, chi_cal, chi_exact, L, suppression_threshold=0.01):
    """Operator decoherence renormalization with mirror-pooled factors and post-selection.

    Works on a single profile (2L,) or on a stack of profiles (n, 2L) at once (used by odr_uncertainty)."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def odr_uncertainty(chi, chi_std, chi_cal, chi_cal_std, chi_exact, L, suppression_threshold=0.01,
                    n_samples=2000, seed=0):
    """1-sigma uncertainty of odr_mitigate(...) by Monte-Carlo propagation of the Estimator stds."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def odr_bias(chi_true, f_phys, f_cal):
    """Systematic shift of the ODR estimate when the physics-circuit factor differs from the calibration factor."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


===== CELL 82 [markdown] tags=[] =====
**Self-test on synthetic data.** We fabricate a "true" $t=8$ profile (the symmetrised QDC bond-40
file), a set of site-dependent suppression factors including two hopeless sites (mirror pair $5
\leftrightarrow 62$ with $f=0.005$), and Gaussian shot noise. Three things must hold:
noise-free recovery is exact (except the two `NaN` sites), the bias formula reproduces the shift
observed when the physics factors are 10 % lower than the calibration factors, and the Monte-Carlo
$\sigma$ matches the scatter of repeated synthetic experiments.


===== CELL 83 [code] tags=[] =====
# Self-test of your ODR implementation on synthetic data (no prompts; it must run without errors)
rng = np.random.default_rng(2026)
chi_true_syn = cp_average(np.loadtxt("reference_data/chi_wave_evolved_sim_L34_maxbond40.txt"))
chi_exact_syn = np.asarray(chi_wave_exact, float)                    # exact t=0 profile from Part 1
f_cal_syn = rng.uniform(0.15, 0.6, 2 * L)
f_cal_syn[[5, 2 * L - 1 - 5]] = 0.005                                # a hopeless mirror pair -> NaN
chi_meas_syn = 1 - f_cal_syn * (1 - chi_true_syn)
chi_cal_syn = 1 - f_cal_syn * (1 - chi_exact_syn)

# (1) noise-free recovery
rec = odr_mitigate(chi_meas_syn, chi_cal_syn, chi_exact_syn, L)
ok = np.isfinite(rec)
assert not ok[5] and not ok[2 * L - 6], "the mirror pair with f = 0.005 must be post-selected away (NaN)"
assert ok.sum() == 2 * L - 2 and np.max(np.abs(rec[ok] - chi_true_syn[ok])) < 1e-10, "noise-free ODR must be exact"
print(f"(1) noise-free recovery: max |error| = {np.max(np.abs(rec[ok] - chi_true_syn[ok])):.1e}, NaN sites = {np.where(~ok)[0].tolist()}")

# (2) bias when the physics circuit is 10 % less noisy than the calibration circuit (f_phys = 0.9 f_cal)
chi_meas_biased = 1 - 0.9 * f_cal_syn * (1 - chi_true_syn)
rec_b = odr_mitigate(chi_meas_biased, chi_cal_syn, chi_exact_syn, L)
predicted = odr_bias(chi_true_syn, 0.9 * f_cal_syn, f_cal_syn)
assert np.nanmax(np.abs((rec_b - chi_true_syn) - predicted)[ok]) < 1e-10, "odr_bias must reproduce the observed shift"
print(f"(2) bias for f_phys/f_cal = 0.9: max |delta chi| = {np.nanmax(np.abs(predicted)):.3f} "
      f"(largest where 1 - chi is largest), formula exact to {np.nanmax(np.abs((rec_b - chi_true_syn) - predicted)[ok]):.1e}")

# (3) Monte-Carlo sigma vs the scatter of 300 repeated synthetic experiments (shot noise std 0.01)
std_syn = np.full(2 * L, 0.01)
sigma_mc = odr_uncertainty(chi_meas_syn, std_syn, chi_cal_syn, std_syn, chi_exact_syn, L, n_samples=2000, seed=0)
reps = np.array([odr_mitigate(rng.normal(chi_meas_syn, std_syn), rng.normal(chi_cal_syn, std_syn), chi_exact_syn, L)
                 for _ in range(300)])
sigma_emp = np.nanstd(reps, axis=0, ddof=1)
ratio = sigma_mc[ok] / sigma_emp[ok]
print(f"(3) MC sigma / empirical sigma: median {np.median(ratio):.2f}, range [{ratio.min():.2f}, {ratio.max():.2f}]; "
      f"sigma grows like 1/f: corr(sigma, 1/f) = {np.corrcoef(sigma_mc[ok], 1 / f_cal_syn[ok])[0, 1]:.2f}")
assert 0.8 < np.median(ratio) < 1.25

fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
ax[0].plot(chi_true_syn, "g-", lw=2, label="true"); ax[0].errorbar(range(2 * L), rec, sigma_mc, fmt="k.", capsize=2, label="ODR (noise-free input, MC sigma)")
ax[0].set_xlabel("site j"); ax[0].set_ylabel(r"$\chi_j$"); ax[0].legend(fontsize=9)
ax[1].plot(f_cal_syn, "o", ms=4, label="f_cal"); ax[1].plot(sigma_mc / 0.01, "s", ms=4, label="sigma_MC / sigma_shot")
ax[1].set_yscale("log"); ax[1].set_xlabel("site j"); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.show()


===== CELL 84 [code] tags=[] =====
# grade 3.1 (synthetic data, fixed seeds)
ff.grade_ex3_1(odr_mitigate, odr_uncertainty, odr_bias)


===== CELL 85 [markdown] tags=[] =====
## 3.2 Your own Pauli twirling and charge post-selection (5 pts)

**Pauli twirling.** The runtime option `twirling.enable_gates=True` inserts, around every two-qubit
gate, a random Pauli pair $P_a\otimes P_b$ *before* the gate and its conjugate *after* it, so that
the ideal circuit is unchanged while the noise is averaged into a Pauli channel. To do it by hand you
need the conjugation rules of CZ. Because ${\rm CZ}=\mathrm{diag}(1,1,1,-1)$ commutes with $Z$ on
either qubit and flips the sign of the $|11\rangle$ component, one finds

$$ {\rm CZ}\,(X\otimes I)\,{\rm CZ} = X\otimes Z,\qquad {\rm CZ}\,(Y\otimes I)\,{\rm CZ} = Y\otimes Z,\qquad
{\rm CZ}\,(Z\otimes I)\,{\rm CZ} = Z\otimes I , $$

and symmetrically for the second qubit. For a product $P_a\otimes P_b$ the rule is: *each $X$ or $Y$
on one qubit multiplies the other qubit's Pauli by $Z$* (up to a phase, e.g. $X\otimes Y \to
(XZ)\otimes(ZY) \propto Y\otimes X$). The full table has 16 entries:

| before | after | before | after | before | after | before | after |
|---|---|---|---|---|---|---|---|
| $II$ | $II$ | $XI$ | $XZ$ | $YI$ | $YZ$ | $ZI$ | $ZI$ |
| $IX$ | $ZX$ | $XX$ | $YY$ | $YX$ | $XY$ | $ZX$ | $IX$ |
| $IY$ | $ZY$ | $XY$ | $YX$ | $YY$ | $XX$ | $ZY$ | $IY$ |
| $IZ$ | $IZ$ | $XZ$ | $XI$ | $YZ$ | $YI$ | $ZZ$ | $ZZ$ |

Since $P'\,{\rm CZ}\,P = {\rm CZ}$ (with $P' = {\rm CZ}\,P\,{\rm CZ}$, Paulis being Hermitian and CZ
self-inverse), sandwiching every CZ with $(P, P')$ leaves the unitary invariant up to a global phase.
The frames must stay in the ISA basis: $X \to$ `x`, $Z \to$ `rz(π)`, $Y \propto Z X \to$ `x` then
`rz(π)`.

**Charge post-selection.** The total charge $Q_{\rm tot} = -\tfrac12\sum_j (Z_j + (-1)^j)
= N_1 - L$ where $N_1$ is the number of qubits in $|1\rangle$; all our states live in $Q_{\rm tot}=0$,
i.e. bitstrings of Hamming weight exactly $L$. Any measured bitstring with a different weight is a
detected error. The same identity gives a cheap **witness** for symmetry-breaking (non-Pauli-like)
noise that needs no post-selection at all:

$$ \sum_j \langle Z_j\rangle = -2\langle Q_{\rm tot}\rangle = 0 \quad\text{(exactly, for every state in the sector)}. $$

Depolarizing noise (any Pauli channel) has no preferred charge direction: it shrinks every
$\langle Z_j\rangle$ and the sum stays close to zero (only the site-to-site *variation* of the suppression
factors leaks in). Amplitude damping pushes every qubit towards $|0\rangle$ and makes the sum
systematically positive, growing with $\gamma\times$(gates per qubit). Twirling removes that offset (up
to the finite number of frames).


===== CELL 86 [code] tags=[] =====
# PROMPT:
#   twirl_circuit(isa_circuit, seed) -> new circuit with random Pauli frames around EVERY cz, re-synthesised into
#       rz/sx/x so the circuit stays ISA (same qubits, same layout metadata: start from isa_circuit.copy_empty_like()).
#   postselect_charge(counts, L) -> the subset of `counts` whose bitstrings have Hamming weight L (charge Q_tot = 0).
#   z_from_counts(counts) -> array <Z_j> (j = qubit index, Qiskit little-endian bitstrings) and
#   charge_witness(counts) -> sum_j <Z_j>  (= -2 <Q_tot>).
# Hint: you may build the CZ conjugation table numerically with qiskit.quantum_info.Pauli/Operator instead of typing it.

# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER


def twirl_circuit(isa_circuit, seed):
    """Return an ISA-equivalent copy of `isa_circuit` with an independent random Pauli frame around every cz."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def postselect_charge(counts, L):
    """Keep only the bitstrings with exactly L ones (total charge zero)."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def z_from_counts(counts):
    """<Z_j> for every qubit j from a counts dictionary (bitstring character n-1-j is qubit j)."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def charge_witness(counts):
    """sum_j <Z_j> = -2 <Q_tot>; zero for any state in the Q_tot = 0 sector."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


===== CELL 87 [markdown] tags=[] =====
**Check and toy demonstration (L = 4, t = 2, Aer).** First the twirled circuit must be *exactly*
equivalent to the original (unitary check on an 8-qubit ISA circuit) and must contain only ISA gates.
Then we run the wavepacket physics circuit under two noise models of similar strength — two-qubit
depolarizing vs. amplitude damping on every CZ — and look at the charge witness and at the fraction of
shots that survive charge post-selection, with and without your twirls.


===== CELL 88 [code] tags=[] =====
# Twirl check + charge-witness toy at L = 4 (no prompts, < 30 s)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

L4 = 4
qc_w4 = prep_wave(L4, TH_OV1, TH_OV3, TH_O11, TH_O22)
qc_w4_phys, _ = evolve_circuits(qc_w4, L4, 2.0, m, g)
pm_basis = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cz"], seed_transpiler=1)
isa4 = pm_basis.run(qc_w4_phys)
tw4 = twirl_circuit(isa4, seed=0)
assert Operator(tw4).equiv(Operator(isa4)), "twirled circuit must equal the original up to a global phase"
assert set(tw4.count_ops()) <= {"rz", "sx", "x", "cz", "barrier"}, "twirled circuit must stay ISA"
print("L=4 physics circuit:", dict(isa4.count_ops()), "-> twirled:", dict(tw4.count_ops()), "| unitary-equivalent: True")
tw68 = twirl_circuit(circuits_all_isa[0], seed=1)
print("68-qubit t=8 ISA circuit:", {k: circuits_all_isa[0].count_ops().get(k, 0) for k in ("cz", "sx", "x", "rz")},
      "-> twirled:", {k: tw68.count_ops().get(k, 0) for k in ("cz", "sx", "x", "rz")},
      "| layout preserved:", tw68.layout.initial_index_layout(filter_ancillas=True) == circuits_all_isa[0].layout.initial_index_layout(filter_ancillas=True))

noise_dep = NoiseModel()
noise_dep.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), "cz")
noise_dep.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["sx", "x"])
noise_ad = NoiseModel()
noise_ad.add_all_qubit_quantum_error(amplitude_damping_error(0.02).tensor(amplitude_damping_error(0.02)), "cz")

def run_counts(circ, noise, shots, seed):
    c = circ.copy(); c.measure_all()
    return AerSimulator(noise_model=noise, seed_simulator=seed).run(c, shots=shots).result().get_counts()

witness_table = {}
for name, noise in [("depolarizing", noise_dep), ("amplitude damping", noise_ad)]:
    counts = run_counts(isa4, noise, 4000, 11)
    kept = sum(postselect_charge(counts, L4).values()) / 4000
    counts_tw = {}
    for s in range(8):                                    # 8 hand-made twirls x 500 shots
        for b, c in run_counts(twirl_circuit(isa4, seed=100 + s), noise, 500, 20 + s).items():
            counts_tw[b] = counts_tw.get(b, 0) + c
    kept_tw = sum(postselect_charge(counts_tw, L4).values()) / 4000
    witness_table[name] = (charge_witness(counts), charge_witness(counts_tw), kept, kept_tw)
    print(f"{name:18s}: witness sum<Z> = {charge_witness(counts):+.3f} (untwirled) {charge_witness(counts_tw):+.3f} (twirled) | "
          f"Q=0 fraction kept = {kept:.3f} / {kept_tw:.3f}")
print("ideal (noise-free) witness:", f"{charge_witness(run_counts(isa4, None, 4000, 3)):+.3f}")


===== CELL 89 [code] tags=[] =====
# grade 3.2 (unitary equivalence + ISA check of twirl_circuit on a hidden circuit; post-selection on hidden counts)
ff.grade_ex3_2(twirl_circuit, postselect_charge)


===== CELL 90 [markdown] tags=[] =====
## 3.3 Noisy rehearsal at L = 6, t = 4 (8 pts)

Before touching the QPU, run the *entire* pipeline — four circuits on a real sub-chain of the
device, Estimator, your ODR, mirror averaging, vacuum subtraction, uncertainties — on a noisy
simulator whose error rates come from the calibration data of `backend.target`, and compare with the
exact answer (12 qubits: a statevector is cheap).

> **Local testing mode is not a noise simulator.** `qiskit_ibm_runtime.EstimatorV2(mode=FakeKingston())`
> silently *ignores* the `twirling`, `dynamical_decoupling` and `resilience` options of Part 2 and
> runs a plain Aer simulation. Do not use it to judge mitigation. We use
> `qiskit_aer.primitives.EstimatorV2` with a **reduced Pauli noise model built by hand**:
> `NoiseModel.from_backend(backend)` (with relaxation) takes minutes for these circuits, while a
> depolarizing + readout model restricted to the 12 qubits we use takes seconds.
>
> How Aer's `EstimatorV2` produces numbers: with a noise model and `method="statevector"` it averages
> the exact expectation value over `shots` random noise *trajectories*, then adds Gaussian noise of
> width `precision` to mimic finite-shot statistics (`stds` is simply that precision). We use 1000
> trajectories and `precision = 1/sqrt(4000)`, the shot-noise level of a 4000-shot experiment.

`reduced_noise_model(backend, chain)` must build, from `backend.target`, for the qubits in `chain`:
a `depolarizing_error(eps_cz, 2)` on `cz` for every chain edge (both qubit orders, `eps_cz` = the
target's CZ error), `depolarizing_error(eps_sx, 1)` on `sx` **and** `x` for every qubit (`eps_sx` = the
target's `sx` error), and a `ReadoutError` from the target's `measure` error (symmetric); no thermal
relaxation. The 12-qubit sub-chain is the contiguous window of your 68-qubit `chain` (Part 2.3) with
the smallest summed CZ error.


===== CELL 91 [code] tags=[] =====
# PROMPT: reduced_noise_model(backend, chain) -> qiskit_aer.noise.NoiseModel (see the text above), and
#         select_subchain(backend, chain, n_qubits=12) -> the contiguous window of `chain` with the smallest summed CZ error.
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


def _cz_error(target, a, b):
    """CZ error of edge (a, b) from the target, whichever direction is listed."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def reduced_noise_model(backend, chain):
    """Hand-built Pauli noise model (depolarizing cz / sx / x + symmetric readout error) for the qubits in `chain`."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def select_subchain(backend, chain, n_qubits=12):
    """Contiguous window of `chain` minimising the summed CZ error of its n_qubits - 1 edges."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


chain_L6 = select_subchain(backend, chain, 12)
noise_model = reduced_noise_model(backend, chain_L6)
print("12-qubit sub-chain:", chain_L6)
print("CZ errors on its edges:", [f"{_cz_error(backend.target, a, b):.4f}" for a, b in zip(chain_L6[:-1], chain_L6[1:])])
print("readout errors:", [f"{backend.target['measure'][(q,)].error:.3f}" for q in chain_L6])
print(noise_model)


===== CELL 92 [code] tags=[] =====
# PROMPT: build the four L=6, t=4 circuits (wave / wave_mitig / vacuum / vacuum_mitig; the vacuum circuit is the
# noise-matched one with the wavepacket layers at angle EPS_VAC), transpile them onto chain_L6 (optimization_level=3,
# initial_layout=chain_L6, seed 42), map the L=6 observables to that layout and run qiskit_aer.primitives.EstimatorV2
# with `noise_model` (method statevector, 1000 trajectories, seed 7, precision 1/sqrt(4000)).
# Names used below: circuits_L6_isa (list of 4), observables_L6_isa, chi_raw_arrays = {"chi_wave", "chi_wave_mitig",
# "chi_vacuum", "chi_vacuum_mitig"} (chi arrays) and chi_std_arrays (same keys, stds).
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

L6, t6 = 6, 4.0
SHOTS_L6, N_TRAJ_L6 = 4000, 1000
qc_wave_init_6 = prep_wave(L6, TH_OV1, TH_OV3, TH_O11, TH_O22)
qc_vacuum_init_6 = prep_wave(L6, TH_OV1, TH_OV3, EPS_VAC, EPS_VAC)          # noise-matched vacuum (Part 0)
qc_wave_6, qc_wave_mitig_6 = evolve_circuits(qc_wave_init_6, L6, t6, m, g)
qc_vacuum_6, qc_vacuum_mitig_6 = evolve_circuits(qc_vacuum_init_6, L6, t6, m, g)
circuits_L6 = [qc_wave_6, qc_wave_mitig_6, qc_vacuum_6, qc_vacuum_mitig_6]
observables_L6 = chiral_condensate_observables(L6)

t0_ = time.time()
# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER
print(f"rehearsal run: {time.time() - t0_:.1f} s")
for c in circuits_L6_isa:
    assert c.layout.initial_index_layout(filter_ancillas=True) == list(chain_L6), "all four circuits must sit on chain_L6"
    assert c.layout.final_index_layout() == list(chain_L6), "no SWAPs expected on a linear chain"
print("CZ counts:", [c.count_ops().get("cz", 0) for c in circuits_L6_isa],
      "| 2q depths:", [c.depth(lambda i: (not getattr(i.operation, "_directive", False)) and len(i.qubits) > 1) for c in circuits_L6_isa])


===== CELL 93 [code] tags=[] =====
# PROMPT: post-process the rehearsal with YOUR functions:
#   X_exact_L6 = exact chi(wave) - chi(vacuum) at t=4 (statevector);  calibration references = exact t=0 profiles
#   X_raw  = raw wave minus raw vacuum (site by site);  X_raw_cp = the same after CP (mirror) averaging of each arm
#   X_mit  = odr_mitigate(wave) - odr_mitigate(vacuum)            sigma_L6 = sqrt(sigma_wave^2 + sigma_vacuum^2) (odr_uncertainty)
#   rmse_raw_L6, rmse_mit_L6 vs X_exact_L6; pulls_L6 = (X_mit - X_exact)/sigma
# Names used below: chi_wave_exact_6, chi_vacuum_exact_6 (exact t=0 profiles), X_exact_L6, X_raw, X_raw_cp, X_mit, sigma_L6, pulls_L6.
# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER
chi_raw_arrays["chi_wave_exact"], chi_raw_arrays["chi_vacuum_exact"] = chi_wave_exact_6, chi_vacuum_exact_6   # calibration references
factors_L6 = {k: (1 - chi_raw_arrays[f"chi_{k}_mitig"]) / (1 - ce) for k, ce in [("wave", chi_wave_exact_6), ("vacuum", chi_vacuum_exact_6)]}
print(f"RMSE vs exact: raw {rmse_raw_L6:.4f} (CP-averaged {rmse(X_raw_cp, X_exact_L6):.4f}) -> ODR {rmse_mit_L6:.4f}  (shot-noise floor ~ {np.nanmedian(sigma_L6):.4f}); "
      f"calibration factors {np.min(factors_L6['wave']):.2f}-{np.max(factors_L6['wave']):.2f}; "
      f"pulls: mean {np.nanmean(pulls_L6):+.2f}, std {np.nanstd(pulls_L6):.2f}, max |pull| {np.nanmax(np.abs(pulls_L6)):.2f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
ax[0].plot(X_exact_L6, "g-o", label="exact"); ax[0].plot(X_raw, "r--s", label="raw (vac. subtracted)")
ax[0].errorbar(range(2 * L6), X_mit, sigma_L6, fmt="k--^", capsize=3, label="your ODR ± MC σ")
ax[0].set_xlabel("site j"); ax[0].set_ylabel(r"$\mathcal{X}_j$"); ax[0].set_title("L=6, t=4 rehearsal"); ax[0].legend(fontsize=8)
ax[1].plot(factors_L6["wave"], "k--o", label="wave"); ax[1].plot(factors_L6["vacuum"], "r--o", label="vacuum")
ax[1].set_xlabel("site j"); ax[1].set_ylabel("calibration factor f"); ax[1].legend(fontsize=8)
ax[2].hist(pulls_L6[np.isfinite(pulls_L6)], bins=np.linspace(-4, 4, 17), color="gray", edgecolor="k")
ax[2].set_xlabel("pull (X_mit - X_exact)/σ"); ax[2].set_title("ODR bias shows up as a shifted/wide pull histogram", fontsize=9)
plt.tight_layout(); plt.show()
np.savez("submission/rehearsal_L6.npz", **chi_raw_arrays, **{k + "_std": v for k, v in chi_std_arrays.items()},
         X_raw=X_raw, X_raw_cp=X_raw_cp, X_mit=X_mit, sigma=sigma_L6, X_exact=X_exact_L6, chain=np.asarray(chain_L6))


===== CELL 94 [code] tags=[] =====
# grade 3.3 (noise model vs the frozen target summary, ODR consistency, RMSE vs the exact L=6 profile)
ff.grade_ex3_3(noise_model, chain_L6, chi_raw_arrays, X_raw, X_mit, odr_mitigate, backend)


===== CELL 95 [markdown] tags=[] =====
<span id="part4"></span>
# Part 4 — Hardware (25 pts)

<div class="alert alert-block alert-danger">
<b>Read before you flip the switch.</b> Every team has a hard cap of <b>180 s of QPU usage</b> for the
whole challenge (Open Plan: 10 min per instance per 28 days, job/batch mode only). The plan is:
canary at t=0 (≈ 4 s) → main t=8 run (≈ 32 s) → one improvement run (≤ 60 s). A job that is
submitted twice, or with the wrong shot count, is usage you will not get back. The cells below only
submit when <code>RUN_ON_HARDWARE = True</code>; with <code>False</code> they load the organizer's
cached dataset (flagged <code>fallback=True</code>, capped at 75 % of the points) so that the notebook
runs offline end-to-end.
</div>

Checklist before `RUN_ON_HARDWARE = True`:

1. Part 3.3 rehearsal passed (ODR beats raw, pulls are $\mathcal O(1)$).
2. `flight_plan` (Part 2.4) predicts ≤ 40 s for canary + main run.
3. `submission/` contains nothing from a previous attempt (a second job id will be flagged).
4. Your account is saved (`QiskitRuntimeService.save_account(...)` *once*, in a terminal — never in this notebook).


===== CELL 96 [code] tags=[] =====
# Master switch, fallback loader and small helpers (no prompts)
RUN_ON_HARDWARE = False          # <- flip to True only after the checklist above; every team has ONE 180 s budget
HW_BACKEND_NAME = backend.name.replace("fake_", "ibm_")   # FakeKingston -> ibm_kingston etc.
USAGE_MODEL = lambda executions: 2.0 + 0.45e-3 * executions   # s, QPU-plan usage model (Part 2.4)


def load_fallback(name):
    """Organizer-released dataset: reference_data/<name> (participant kit) or organizer/fallback_data/<name>."""
    for folder in ("reference_data", os.path.join("organizer", "fallback_data")):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            out = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
            print(f"loaded fallback dataset {path}: job {out.get('job_id')} on {out.get('backend')} ({out.get('note', '')[:80]}...)")
            return out
    raise FileNotFoundError(f"{name} not found in reference_data/ or organizer/fallback_data/ — it is released on Day 2 to teams "
                            "without a returned job; until then set RUN_ON_HARDWARE = True or wait.")


def with_twirling(options, num_randomizations, shots_per_randomization):
    """Copy of the Part-2 Estimator options with a different twirling budget (works for dicts and EstimatorOptions)."""
    opts = copy.deepcopy(options)
    if isinstance(opts, dict):
        tw = dict(opts.get("twirling", {}) or {})
        tw.update(enable_gates=True, enable_measure=True, num_randomizations=num_randomizations,
                  shots_per_randomization=shots_per_randomization)
        opts["twirling"] = tw
    else:
        opts.twirling.enable_gates = True
        opts.twirling.enable_measure = True
        opts.twirling.num_randomizations = num_randomizations
        opts.twirling.shots_per_randomization = shots_per_randomization
    return opts


# reference profile for scoring: organizer bond-64 MPS (bond-40 QDC files as a last resort)
WINDOW = np.arange(25, 43)                       # W = {25, ..., 42}
_mps_path = os.path.join("reference_data", "mps_reference_L34.npz")
if os.path.exists(_mps_path):
    _mps = np.load(_mps_path)
    X_ref_bd64 = _mps["chi_wave_t8_bd64"] - _mps["chi_vacuum_t8_bd64"]
    X_ref_label = "MPS bond 64"
else:
    warnings.warn("reference_data/mps_reference_L34.npz missing -> using the QDC bond-40 files as reference")
    X_ref_bd64 = (np.loadtxt("reference_data/chi_wave_evolved_sim_L34_maxbond40.txt")
                  - np.loadtxt("reference_data/chi_vacuum_evolved_sim_L34_maxbond40.txt"))
    X_ref_label = "QDC MPS bond 40"
layout_used = list(circuits_all_isa[0].layout.initial_index_layout(filter_ancillas=True))
print(f"RUN_ON_HARDWARE = {RUN_ON_HARDWARE}; hardware target would be {HW_BACKEND_NAME}; reference = {X_ref_label}; "
      f"X_ref peaks {X_ref_bd64[[31, 32, 35, 36]].round(3).tolist()}, dip {X_ref_bd64[[33, 34]].round(3).tolist()}")


===== CELL 97 [markdown] tags=[] =====
## 4.1 Canary at t = 0 (3 pts, ≈ 4 s of usage)

A canary is a cheap job that tells you whether the chain you chose is alive *today*: the two $t=0$
circuits (wavepacket and vacuum, ~60 CZ each after transpilation) have exactly known
$\langle Z_j\rangle$, so the **retention** $r_j = \langle Z_j\rangle_{\rm meas}/\langle Z_j\rangle_{\rm exact} =
(1-\chi_{j,\rm meas})/(1-\chi_{j,\rm exact})$ measures readout + short-circuit fidelity per qubit, and
the **contrast** of the hadron at sites 33, 34 ($\mathcal X_{33,34}(t{=}0) \approx 1.43$) tells you
whether the central qubits — the ones that carry the whole signal at $t=8$ — are usable.

Pass/fail rule (also used by the organizers): the per-site retention is scored on the **vacuum arm**
(the state closest to what the calibration circuits return to; the wavepacket arm additionally
carries the hadron and is used for the contrast). **Pass** if at most 2 qubits have $r_j < 0.4$ *and*
the contrast retention is ≥ 0.15; otherwise **re-select the chain (Part 2.3) before the main run**
(a qubit that fails only in the wavepacket arm is a warning to note in the report). Budget: 2 PUBs ×
16 twirls × 128 shots = 4,096 executions ≈ 4 s.


===== CELL 98 [code] tags=['hardware'] =====
# PROMPT: canary. Transpile qc_wave_init / qc_vacuum_init onto layout_used (optimization_level 3, seed 42), build
# 2 PUBs with the chi observables, and (if RUN_ON_HARDWARE) run them with the Part-2 options at 16 twirls x 128 shots.
# Otherwise load canary_fallback.npz. Then compute the per-site retention for both arms, the contrast retention,
# apply the pass/fail rule (<= 2 flagged sites with vacuum-arm retention r < 0.4 AND contrast retention >= 0.15).
# Names used below: retention_wave, retention_vacuum, X0_exact, X0_meas (t=0 vacuum-subtracted profiles), contrast_retention
# (measured/exact max of X0 over the two central sites L-1, L), flagged_sites (vacuum arm), flagged_sites_wave,
# flagged_qubits (= canary_layout[flagged_sites]), canary_pass (bool), verdict (str).
CANARY_TWIRLS, CANARY_SHOTS = 16, 128
pm_layout = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=layout_used, seed_transpiler=42)
canary_circuits = [pm_layout.run(qc_wave_init), pm_layout.run(qc_vacuum_init)]
canary_observables = [o.apply_layout(canary_circuits[0].layout) for o in observables]
canary_pubs = [(c, canary_observables) for c in canary_circuits]
print("canary circuits: CZ =", [c.count_ops().get("cz", 0) for c in canary_circuits],
      "| predicted usage ~", f"{USAGE_MODEL(2 * CANARY_TWIRLS * CANARY_SHOTS):.1f} s")

if RUN_ON_HARDWARE:
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimatorV2
    service = QiskitRuntimeService()                                   # saved account; never paste tokens here
    hw_backend = service.backend(HW_BACKEND_NAME)
    assert all(hw_backend.target.instruction_supported(inst.operation.name, tuple(c.find_bit(q).index for q in inst.qubits))
               for c in canary_circuits for inst in c.data if inst.operation.name != "barrier"), "re-transpile against hw_backend.target"
    canary_estimator = RuntimeEstimatorV2(mode=hw_backend, options=with_twirling(estimator_options, CANARY_TWIRLS, CANARY_SHOTS))
    canary_job = canary_estimator.run(canary_pubs)
    print("canary job id:", canary_job.job_id())
    with open("submission/canary_job_id.txt", "w") as fh:
        fh.write(canary_job.job_id())
    canary_res = canary_job.result()
    canary_evs = np.array([np.asarray(r.data.evs, float) for r in canary_res])
    canary_stds = np.array([np.asarray(r.data.stds, float) for r in canary_res])
    canary_layout = np.asarray(layout_used)
    job_info_canary = {"job_id": canary_job.job_id(), "backend": hw_backend.name, "usage_s": float(canary_job.usage() or 0.0),
                       "num_randomizations": CANARY_TWIRLS, "shots_per_randomization": CANARY_SHOTS,
                       "submitted": str(canary_job.creation_date), "fallback": False}
else:
    fb = load_fallback("canary_fallback.npz")
    canary_evs, canary_stds, canary_layout = fb["evs"], fb["stds"], fb["layout"]
    job_info_canary = {"job_id": fb["job_id"], "backend": fb["backend"], "usage_s": float(fb["usage_s"]),
                       "usage_estimated": bool(fb.get("usage_estimated", True)), "shots": int(fb.get("shots", 0)),
                       "num_randomizations": 1, "shots_per_randomization": int(fb.get("shots", 0)),
                       "submitted": str(fb.get("submitted", "")), "fallback": True}

# BEGIN ANSWER
# YOUR CODE HERE
# END ANSWER
job_info_canary["layout"] = [int(q) for q in canary_layout]
canary_result = {"chi_wave": canary_evs[0], "chi_vacuum": canary_evs[1], "evs": canary_evs, "stds": canary_stds,
                 "layout": np.asarray(canary_layout), "retention_wave": retention_wave, "retention_vacuum": retention_vacuum,
                 "contrast_retention": contrast_retention, "flagged_sites": flagged_sites, "flagged_sites_wave": flagged_sites_wave,
                 "flagged_qubits": flagged_qubits, "pass": bool(canary_pass), "verdict": verdict,
                 "fallback": bool(job_info_canary["fallback"])}
print(f"canary on {job_info_canary['backend']} (job {job_info_canary['job_id']}, usage {job_info_canary['usage_s']:.0f} s"
      f"{' est.' if job_info_canary.get('usage_estimated') else ''}): median retention wave {np.median(retention_wave):.2f} / "
      f"vacuum {np.median(retention_vacuum):.2f}, contrast retention {contrast_retention:.2f}, "
      f"flagged sites {flagged_sites} -> qubits {flagged_qubits} (wave arm: {flagged_sites_wave})  =>  {verdict}")
fig, ax = plt.subplots(1, 2, figsize=(15, 3.6))
ax[0].plot(retention_wave, "k--o", label="wave"); ax[0].plot(retention_vacuum, "r--o", label="vacuum")
ax[0].axhline(0.4, color="gray", ls=":"); ax[0].set_xlabel("site j"); ax[0].set_ylabel("retention r_j"); ax[0].legend(fontsize=8)
ax[1].plot(X0_exact, "g-o", label="exact t=0"); ax[1].plot(X0_meas, "k--s", label="measured")
ax[1].set_xlabel("site j"); ax[1].set_ylabel(r"$\mathcal{X}_j(t=0)$"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
np.savez("submission/canary_t0.npz", evs=canary_evs, stds=canary_stds, layout=np.asarray(canary_layout),
         retention_wave=retention_wave, retention_vacuum=retention_vacuum, flagged_sites=np.asarray(flagged_sites, int),
         verdict=verdict, job_info=json.dumps(job_info_canary))


===== CELL 99 [code] tags=[] =====
# grade 4.1
ff.grade_ex4_1(canary_result, job_info_canary)


===== CELL 100 [markdown] tags=[] =====
## 4.2 Main run at t = 8 (18 pts — the leaderboard)

4 PUBs (`circuits_all_isa` × `observables_isa`) × 64 twirls × 256 shots = 65,536 executions ≈ 32 s.
The runtime returns `evs` and `stds` per observable; **all mitigation happens in your code**:

$\chi^{\rm mit}$ = `odr_mitigate` (calibration reference = the exact $t=0$ arrays), $\sigma$ from
`odr_uncertainty`, $\mathcal X^{\rm mit}_j = \chi^{\rm mit,wave}_j - \chi^{\rm mit,vac}_j$,
$\sigma_j = \sqrt{\sigma^{\rm wave\,2}_j + \sigma^{\rm vac\,2}_j}$; the raw profile is the CP-averaged,
vacuum-subtracted one.

The grader recomputes $\hat{\mathcal X}$ from your submitted `evs` with **your** `odr_mitigate` and
requires it to equal `X_mit` to $10^{-6}$; then (SPEC): $\mathrm{RMSE}_W$ on $W=\{25..42\}$
vs the bond-64 MPS reference, contrast $C = \overline{\hat{\mathcal X}}_{31,32,35,36} -
\overline{\hat{\mathcal X}}_{33,34}$ vs $C_{\rm ref}=0.515$, a dip test, and a quiet-region test
$\mathrm{median}_{j\notin W}|\hat{\mathcal X}_j| \le 0.05$. The cell below prints a *preview* of that score.


===== CELL 101 [code] tags=['hardware'] =====
# PROMPT: main run. If RUN_ON_HARDWARE: submit the 4 PUBs with the Part-2 options at 64 twirls x 256 shots, save the job id to
# submission/job_id.txt, collect evs (4, 2L) and stds (4, 2L). Otherwise load hardware_fallback.npz.
# Then write postprocess_run(evs, stds) -> dict(X_raw, X_mit, sigma, sigma_raw, chi_mit_wave, chi_mit_vacuum, factors_wave,
# factors_vacuum) using YOUR odr_mitigate / odr_uncertainty with chi_wave_exact / chi_vacuum_exact as calibration references
# (sigma_raw = shot-noise sigma of the CP-averaged raw profile, propagated from the Estimator stds).
MAIN_TWIRLS, MAIN_SHOTS = 64, 256
main_pubs = [(c, observables_isa) for c in circuits_all_isa]
print("flight plan:", flight_plan, "| predicted usage ~", f"{USAGE_MODEL(4 * MAIN_TWIRLS * MAIN_SHOTS):.1f} s")

if RUN_ON_HARDWARE:
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimatorV2
    service = QiskitRuntimeService()
    hw_backend = service.backend(HW_BACKEND_NAME)
    main_estimator = RuntimeEstimatorV2(mode=hw_backend, options=with_twirling(estimator_options, MAIN_TWIRLS, MAIN_SHOTS))
    main_job = main_estimator.run(main_pubs)
    print("main job id:", main_job.job_id())
    with open("submission/job_id.txt", "w") as fh:
        fh.write(main_job.job_id())
    main_res = main_job.result()
    hw_evs = np.array([np.asarray(r.data.evs, float) for r in main_res])
    hw_stds = np.array([np.asarray(r.data.stds, float) for r in main_res])
    hw_layout = np.asarray(layout_used)
    job_info = {"job_id": main_job.job_id(), "backend": hw_backend.name, "usage_s": float(main_job.usage() or 0.0),
                "num_randomizations": MAIN_TWIRLS, "shots_per_randomization": MAIN_SHOTS,
                "submitted": str(main_job.creation_date), "fallback": False}
else:
    fb = load_fallback("hardware_fallback.npz")
    hw_evs, hw_stds, hw_layout = fb["evs"], fb["stds"], fb["layout"]
    job_info = {"job_id": fb["job_id"], "backend": fb["backend"], "usage_s": float(fb["usage_s"]),
                "usage_estimated": bool(fb.get("usage_estimated", True)),
                "num_randomizations": int(fb["num_randomizations"]), "shots_per_randomization": int(fb["shots_per_randomization"]),
                "submitted": str(fb.get("submitted", "")), "fallback": True}
    with open("submission/job_id.txt", "w") as fh:
        fh.write(f"{fb['job_id']}  (organizer fallback dataset, {fb['backend']})")
assert hw_evs.shape == (4, 2 * L) and hw_stds.shape == (4, 2 * L)
job_info["layout"] = [int(q) for q in hw_layout]
PUB_NAMES = ("chi_wave", "chi_wave_mitig", "chi_vacuum", "chi_vacuum_mitig")


def as_hardware_result(evs, stds, post, layout, fallback):
    """hardware_result dict: the 4 PUB arrays (+ stds), exact t=0 references, and the post-processed profiles."""
    out = {k: np.asarray(e, float) for k, e in zip(PUB_NAMES, evs)}
    out.update({k + "_std": np.asarray(s, float) for k, s in zip(PUB_NAMES, stds)})
    out.update(evs=np.asarray(evs, float), stds=np.asarray(stds, float), layout=np.asarray(layout), fallback=bool(fallback),
               chi_exact_wave=np.asarray(chi_wave_exact, float), chi_exact_vacuum=np.asarray(chi_vacuum_exact, float),
               chi_wave_exact=np.asarray(chi_wave_exact, float), chi_vacuum_exact=np.asarray(chi_vacuum_exact, float))
    out.update({k: post[k] for k in ("X_raw", "X_mit", "sigma", "sigma_raw", "factors_wave", "factors_vacuum")})
    return out


def postprocess_run(evs, stds, threshold=0.01):
    """Raw and ODR-mitigated vacuum-subtracted profiles (+ MC uncertainties) from the 4 PUB results."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def leaderboard_preview(X_hat, sigma, X_ref, W=WINDOW, C_ref=0.515):
    """Preview of the ex 4.2 metric (SPEC formula). The grader is authoritative."""
    X_hat = np.asarray(X_hat, float); err = X_hat - X_ref
    inW = np.isfinite(X_hat[W]); n_nan = int((~inW).sum())
    rmse_W = float(np.sqrt(np.mean(err[W][inW] ** 2)))
    peaks, dip = X_hat[[31, 32, 35, 36]], X_hat[[33, 34]]
    C = float(np.nanmean(peaks) - np.nanmean(dip))
    outside = np.setdiff1d(np.arange(2 * L), W)
    quiet = float(np.nanmedian(np.abs(X_hat[outside])))
    cover = float(np.mean(np.abs(err[W][inW]) <= 2 * np.sqrt(np.asarray(sigma)[W][inW] ** 2 + 0.003 ** 2)))
    pts = (10 * np.clip(1 - rmse_W / 0.25, 0, 1) + 4 * np.clip(1 - abs(C - C_ref) / 0.35, 0, 1)
           + 2 * float(C >= 0.20 and np.nanmin(peaks) - np.nanmax(dip) >= 0.25) + 2 * float(quiet <= 0.05))
    pts *= max(0.0, 1 - 0.05 * n_nan)
    return {"RMSE_W": rmse_W, "contrast": C, "quiet": quiet, "coverage": cover, "n_nan_W": n_nan, "points_preview": float(pts)}


hw = postprocess_run(hw_evs, hw_stds)
X_raw_hw, X_mit_hw, sigma_hw = hw["X_raw"], hw["X_mit"], hw["sigma"]
prev_raw, prev_mit = leaderboard_preview(X_raw_hw, hw["sigma_raw"], X_ref_bd64), leaderboard_preview(X_mit_hw, sigma_hw, X_ref_bd64)
n_small = int(np.sum(np.minimum(hw["factors_wave"], hw["factors_vacuum"])[WINDOW] < 0.9))
print(f"job {job_info['job_id']} on {job_info['backend']}, usage {job_info['usage_s']:.0f} s{' (estimated)' if job_info.get('usage_estimated') else ''}, fallback={job_info['fallback']}")
print(f"raw : RMSE_W {prev_raw['RMSE_W']:.3f}  contrast {prev_raw['contrast']:.3f}  quiet {prev_raw['quiet']:.3f}  -> preview {prev_raw['points_preview']:.1f}/18")
print(f"ODR : RMSE_W {prev_mit['RMSE_W']:.3f}  contrast {prev_mit['contrast']:.3f}  quiet {prev_mit['quiet']:.3f}  coverage {prev_mit['coverage']:.2f}  "
      f"NaN in W {prev_mit['n_nan_W']}  -> preview {prev_mit['points_preview']:.1f}/18{' (x0.75 fallback cap)' if job_info['fallback'] else ''}; "
      f"factors < 0.9 on {n_small}/{len(WINDOW)} window sites (gate: >= 10)")

fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
ax[0].plot(X_ref_bd64, "g-o", lw=2, label=f"{X_ref_label} reference")
ax[0].errorbar(range(2 * L), X_raw_hw, hw["sigma_raw"], fmt="r--s", capsize=3, lw=1, label="hardware raw (CP avg, vac. subtracted)")
ax[0].errorbar(range(2 * L), X_mit_hw, sigma_hw, fmt="k--^", capsize=3, lw=1, label="your ODR ± MC σ")
ax[0].axvspan(WINDOW[0] - 0.5, WINDOW[-1] + 0.5, color="gold", alpha=0.15, label="scoring window W")
ax[0].set_ylabel(r"$\mathcal{X}_j$"); ax[0].set_title(f"t = 8 on {job_info['backend']}"); ax[0].legend(loc="upper left", fontsize=9)
ax[1].plot(hw["factors_wave"], "k--o", label="wave"); ax[1].plot(hw["factors_vacuum"], "r--o", label="vacuum")
ax[1].axhline(0.9, color="gray", ls=":"); ax[1].set_ylabel("calibration factor"); ax[1].set_xlabel("staggered site j"); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

hardware_result = as_hardware_result(hw_evs, hw_stds, hw, hw_layout, job_info["fallback"])
hardware_result["preview"] = prev_mit
np.savez("submission/hardware_t8.npz", **{k: v for k, v in hardware_result.items() if isinstance(v, np.ndarray)},
         fallback=bool(job_info["fallback"]), job_info=json.dumps(job_info))


===== CELL 102 [code] tags=[] =====
# grade 4.2 (recomputes X_hat from evs with YOUR odr_mitigate, then applies the leaderboard metric)
ff.grade_ex4_2(hardware_result, job_info, odr_mitigate)


===== CELL 103 [markdown] tags=[] =====
## 4.3 Improvement run (4 pts, ≤ 60 s of usage)

Pick **one** modification, predict its effect, run it, and compare with the main run using
*z-scores* — a number without an uncertainty is not a result. Menu (others welcome, but write the
rationale first):

| option | what changes | budget hint |
|---|---|---|
| second chain | `select_chain` with the flagged qubits of 4.1 excluded; average the two runs per site | 4 PUBs × 32 × 256 |
| matched calibration | replace `qc_*_mitig` by `evolve_circuits_matched(...)` of 2.2 (same gate count as physics) | 2 PUBs (the two new calibration circuits) |
| dt = 0.5 | 16 Trotter steps: half the Trotter error, twice the CZ count — compare with its **own** MPS reference (Bonus B1) | 4 PUBs × 32 × 256 |
| ZNE | noise factors (1, 3) by gate folding, linear extrapolation of $\mathcal X$; ODR unchanged | 4 PUBs × 2 factors × 16 × 256 |
| hand-twirled PUBs | 16 `twirl_circuit` copies per circuit with `twirling.enable_gates=False`; post-select with `postselect_charge` via `SamplerV2` | 64 PUBs × 256 shots |
| fractional gates | `service.backend(name, use_fractional_gates=True)` + the `rzz` barbell of Bonus B3; fewer 2q gates | 4 PUBs × 32 × 256 |

Report `improvement = {"strategy", "rationale" (≥ 150 characters), "usage_s" (of the improvement job, ≤ 60),
"baseline": {...}, "improved": {...}, "z_score", "z_site", "z_contrast", "rmse_W"}` where each run dict is a
`hardware_result`-like dict (the four `chi_*` arrays, `X_mit`, `sigma`, …). Two kinds of z-score:
per site $z_j = (\mathcal X^{\rm imp}_j - \mathcal X^{\rm base}_j)/\sqrt{\sigma^{\rm imp\,2}_j + \sigma^{\rm base\,2}_j}$,
and the headline `z_score` $= (\mathrm{RMSE}_W^{\rm base} - \mathrm{RMSE}_W^{\rm imp}) / \sqrt{s_{\rm base}^2 + s_{\rm imp}^2}$
with $s$ the uncertainty of each $\mathrm{RMSE}_W$ obtained by linear propagation of the $\sigma_j$
($\partial\,\mathrm{RMSE}_W/\partial \mathcal X_j = (\mathcal X_j - \mathcal X^{\rm ref}_j)/(n\,\mathrm{RMSE}_W)$).
An improvement counts when `z_score` ≥ 2 *and* $\mathrm{RMSE}_W$ went down; a documented null result earns half.


===== CELL 104 [code] tags=['hardware'] =====
# PROMPT: improvement run. Fill `improvement` (see the table above). If RUN_ON_HARDWARE, submit YOUR modified PUBs here
# (<= 60 s of usage) and post-process them with postprocess_run; otherwise the cell demonstrates the z-score comparison on two
# cached ibm_kingston strategies from reference_data/hardware_ibm_kingston_2026-07-25.npz (an ILLUSTRATION, not an improvement
# you made: baseline = 'twirl_dd' (gate+measure twirling + DD XY4, the main-run options), improved = 'trex' (same + TREX
# readout mitigation), plus the 'odr' set, which repeats the twirl_dd options in a second job and thus shows the *null* spread).
def cached_strategy(name, T=8):
    """chi evs/stds (4, 2L) for [wave, wave_mitig, vacuum, vacuum_mitig] of one cached ibm_kingston job."""
    d = np.load("reference_data/hardware_ibm_kingston_2026-07-25.npz")
    prov = json.load(open("reference_data/hardware_ibm_kingston_2026-07-25_provenance.json"))[f"{name}__T{T}"]
    sign = (-1.0) ** np.arange(2 * L)
    evs = np.array([sign * d[f"{name}__T{T}__z_{k}"] + 1 for k in ("wave", "mitig_wave", "vacuum", "mitig_vacuum")])
    stds = np.array([d[f"{name}__T{T}__z_{k}_std"] for k in ("wave", "mitig_wave", "vacuum", "mitig_vacuum")])
    tw = prov["options"]["twirling"]
    return {"evs": evs, "stds": stds, "job_id": prov["job_id"], "backend": prov["backend"],
            "usage_s": USAGE_MODEL(4 * tw["num_randomizations"] * tw["shots_per_randomization"]), "usage_estimated": True,
            "options": prov["options"]}


def rmse_W_with_sigma(run, W=WINDOW):
    """RMSE_W of a run and its 1-sigma uncertainty by linear propagation of the per-site sigmas."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def compare_runs(base, imp):
    """Headline z_score (RMSE_W change), z-scores per site and for the contrast; RMSE_W (+- sigma) of both runs."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


if RUN_ON_HARDWARE:
    # Example skeleton for the "matched calibration" option; adapt for your choice.
    # from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimatorV2
    # qc_w_m, qc_v_m = evolve_circuits_matched(...); isa_m = [pm_layout.run(c) for c in (qc_w_m, qc_v_m)]
    # job_imp = RuntimeEstimatorV2(mode=hw_backend, options=with_twirling(estimator_options, 32, 256)).run([(c, observables_isa) for c in isa_m])
    # evs_imp = np.array([hw_evs[0], job_imp.result()[0].data.evs, hw_evs[2], job_imp.result()[1].data.evs]); stds_imp = ...
    # imp = as_hardware_result(evs_imp, stds_imp, postprocess_run(evs_imp, stds_imp), layout_used, False)
    # z_score, z_site, z_contrast, rmse_W_both = compare_runs(hardware_result, imp)
    # improvement = {"strategy": ..., "rationale": ..., "usage_s": float(job_imp.usage()), "baseline": hardware_result, "improved": imp,
    #                "z_score": z_score, "z_site": z_site, "z_contrast": z_contrast, "rmse_W": rmse_W_both}
    raise NotImplementedError("fill in your improvement run (and keep its usage <= 60 s)")
else:
    runs = {}
    for name in ("twirl_dd", "trex", "odr"):
        r = cached_strategy(name)
        post = postprocess_run(r["evs"], r["stds"])
        r.update(as_hardware_result(r["evs"], r["stds"], post, hw_layout, fallback=True))
        runs[name] = r
    z_score, z_site, z_contrast, rmse_W_both = compare_runs(runs["twirl_dd"], runs["trex"])
    z_null_score, z_null, z_null_contrast, _ = compare_runs(runs["twirl_dd"], runs["odr"])
    improvement = {
        "strategy": "ILLUSTRATION (cached ibm_kingston 2026-07-25): twirl+DD baseline vs twirl+DD+TREX readout mitigation",
        "rationale": "Illustration on cached data, not a run we made. Baseline: gate+measure twirling with XY4 dynamical decoupling "
                     "(the main-run options). Change: TREX measurement-error mitigation (resilience measure_mitigation=True, 32 "
                     "measurement-noise randomizations) applied by the runtime before our ODR. Expected effect: readout bias is removed "
                     "before the calibration factors are formed, so the ODR factors move towards 1 on the qubits with the worst "
                     "measurement error (readout errors up to 5 % on this chain) and the mitigated profile should change by more than "
                     "the shot-noise spread there; the vacuum-subtracted profile is expected to sharpen at the central dip.",
        "illustration": True,
        "baseline": runs["twirl_dd"], "improved": runs["trex"],
        "z_score": z_score, "z_site": z_site, "z_contrast": z_contrast, "rmse_W": rmse_W_both,
        "null_comparison": {"z_score": z_null_score, "z_site": z_null, "z_contrast": z_null_contrast,
                            "job_ids": (runs["twirl_dd"]["job_id"], runs["odr"]["job_id"])},
        "usage_s": runs["trex"]["usage_s"], "usage_note": "cached organizer jobs: 100 twirls x 1000 shots each, ~182 s (est.) — exceeds the 60 s cap of a real improvement run",
    }
    print(f"RMSE_W: baseline {rmse_W_both['baseline']:.3f} +- {rmse_W_both['sigma_baseline']:.3f} -> improved {rmse_W_both['improved']:.3f} "
          f"+- {rmse_W_both['sigma_improved']:.3f}: z_score = {z_score:+.2f}; contrast z = {z_contrast:+.2f}; "
          f"|z_site| in W: median {np.nanmedian(np.abs(z_site[WINDOW])):.2f}, max {np.nanmax(np.abs(z_site[WINDOW])):.2f}")
    print(f"null comparison (two jobs, identical options): z_score = {z_null_score:+.2f}, |z_site| in W median {np.nanmedian(np.abs(z_null[WINDOW])):.2f}, "
          f"max {np.nanmax(np.abs(z_null[WINDOW])):.2f}, contrast z = {z_null_contrast:+.2f}  <- what 'no change' looks like")
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.6))
    ax[0].errorbar(range(2 * L), runs["twirl_dd"]["X_mit"], runs["twirl_dd"]["sigma"], fmt="k--o", capsize=2, ms=3, label="baseline (twirl+DD)")
    ax[0].errorbar(range(2 * L), runs["trex"]["X_mit"], runs["trex"]["sigma"], fmt="b--s", capsize=2, ms=3, label="improved (+TREX)")
    ax[0].plot(X_ref_bd64, "g-", lw=2, label="reference"); ax[0].set_xlabel("site j"); ax[0].set_ylabel(r"$\mathcal{X}_j$"); ax[0].legend(fontsize=8)
    ax[1].plot(z_site, "b-s", ms=3, label=f"improved vs baseline (z_score {z_score:+.1f})"); ax[1].plot(z_null, "k:o", ms=3, label=f"null: repeat job (z_score {z_null_score:+.1f})")
    ax[1].axhspan(-2, 2, color="gray", alpha=0.15); ax[1].set_xlabel("site j"); ax[1].set_ylabel("z-score"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()
np.savez("submission/improvement.npz", strategy=improvement["strategy"], rationale=improvement["rationale"], usage_s=improvement["usage_s"],
         base_evs=improvement["baseline"]["evs"], base_stds=improvement["baseline"]["stds"], base_X=improvement["baseline"]["X_mit"],
         imp_evs=improvement["improved"]["evs"], imp_stds=improvement["improved"]["stds"], imp_X=improvement["improved"]["X_mit"],
         z_score=improvement["z_score"], z_site=improvement["z_site"], z_contrast=improvement["z_contrast"],
         illustration=bool(improvement.get("illustration", False)))


===== CELL 105 [code] tags=[] =====
# grade 4.3
ff.grade_ex4_3(improvement, odr_mitigate)


===== CELL 106 [markdown] tags=[] =====
<span id="part5"></span>
# Part 5 — Error budget and defence (10 pts, judged)

Fill the one-page table below in your report (`submission/report.md`, ≤ 1 page + figures). Every
row must cite the cell that produced the number, and the last column says whether the effect is
removed by vacuum subtraction / ODR, merely estimated, or ignored.

| source | quantity | value | where measured | removed / estimated / ignored |
|---|---|---|---|---|
| state preparation | $1-F$ of the 2-step SC-ADAPT-VQE vacuum (L=8) | *…* | 1.3 | ignored (same in wave and vacuum) |
| Hamiltonian truncation | max shift of $\mathcal X_j$, full vs range-1 electric term (L=8, t=4) | *…* | 1.2 | estimated |
| Trotter ($dt=1$) | max error at L=8, t=4 (proxy for L=34) | *…* | 1.4 / B1 | estimated (Richardson in B1) |
| MPS reference | max $\lvert\mathcal X^{(40)}_j - \mathcal X^{(64)}_j\rvert$ at t=8 | *…* | 1.5 | estimated |
| shot noise | median $\sigma_j$ of the mitigated profile in $W$ | *…* | 4.2 | propagated (MC) |
| ODR bias | rehearsal residual RMSE (L=6) and analytic shift for a 10 % factor mismatch | *…* | 3.3 / 3.1 | partly removed by subtraction |
| hardware | $\mathrm{RMSE}_W$, contrast $C$ (raw → ODR) | *…* | 4.2 | — |
| canary | flagged qubits, contrast retention at t=0 | *…* | 4.1 | chain selection |

The cell below fills the table from the variables of this notebook (missing variables are printed as
`n/a` — go back and compute them).


===== CELL 107 [code] tags=[] =====
# Error-budget table from the notebook variables (no prompts; extend it in your report)
def _get(name, default=None):
    return globals().get(name, default)

def _fmt(x, nd=4):
    try:
        if x is None:
            return "n/a"
        if isinstance(x, dict):
            return ", ".join(f"{k}: {_fmt(v, nd)}" for k, v in x.items())
        x = np.asarray(x, float)
        return f"{np.nanmax(np.abs(x)):.{nd}f}" if x.ndim else f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

_vqe = _get("vqe_fidelity")
_trot = _get("trotter_table")
_trot_41 = None if not isinstance(_trot, dict) else next((v for k, v in _trot.items() if tuple(np.round(k, 3)) == (4.0, 1.0)), None)
_mps_err = _get("mps_err")
_mps40 = _mps_err.get(40) if isinstance(_mps_err, dict) else None
if _mps40 is None and X_ref_label == "MPS bond 64":
    _mps40 = float(np.max(np.abs(np.loadtxt("reference_data/chi_wave_evolved_sim_L34_maxbond40.txt")
                                 - np.loadtxt("reference_data/chi_vacuum_evolved_sim_L34_maxbond40.txt") - X_ref_bd64)))
_peak = int(WINDOW[np.argmax(X_ref_bd64[WINDOW])])
_chi_peak = float(np.asarray(_mps["chi_wave_t8_bd64"])[_peak]) if X_ref_label == "MPS bond 64" else 0.5
_bias10 = float(odr_bias(_chi_peak, 0.9, 1.0))
rows = [
    ("state preparation", "1 - F (L=8)", _fmt(None if not isinstance(_vqe, dict) else 1 - _vqe.get(8, np.nan)), "1.3", "ignored"),
    ("Hamiltonian truncation", "max |dX| full vs range-1 (L=8, t=4)", _fmt(_get("truncation_shift")), "1.2", "estimated"),
    ("Trotter dt=1", "max err (L=8, t=4)", _fmt(_trot_41), "1.4 / B1", "estimated"),
    ("MPS reference", "max |X(40) - X(64)| at t=8", _fmt(_mps40), "1.5", "estimated"),
    ("shot noise", "median sigma_j in W (ODR)", _fmt(np.nanmedian(sigma_hw[WINDOW])), "4.2", "propagated (MC)"),
    ("ODR bias", f"rehearsal RMSE (L=6) {rmse_mit_L6:.4f}; 10 % mismatch at site {_peak}: {_bias10:+.3f} in chi", "", "3.3 / 3.1", "partly removed"),
    ("hardware", f"RMSE_W {prev_raw['RMSE_W']:.3f} -> {prev_mit['RMSE_W']:.3f}; C {prev_raw['contrast']:.3f} -> {prev_mit['contrast']:.3f}", "", "4.2", "-"),
    ("canary", f"flagged qubits {canary_result['flagged_qubits']}, contrast retention {canary_result['contrast_retention']:.2f}", "", "4.1", "chain selection"),
]
table = "| source | quantity | value | cell | status |\n|---|---|---|---|---|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
try:
    from IPython.display import Markdown, display
    display(Markdown(table))
except Exception:
    print(table)
with open("submission/error_budget.md", "w") as fh:
    fh.write(table + "\n")


===== CELL 108 [markdown] tags=[] =====
### Report and viva

* `submission/report.md` (≤ 1 page + up to 4 figures): the table above with your numbers, the ODR
  bias derivation in your own words (3.1), what the canary told you and what you did about it, the
  improvement-run hypothesis and its z-score verdict, and the honest answer to *"which number in your
  final plot do you trust least, and why?"*.
* **8-minute pitch + 7-minute viva** (10 pts, judges): one team member presents, *another* answers
  the questions. Expect: "walk me through `odr_mitigate` line by line", "why does amplitude damping
  break ODR and what does the witness show", "what would you change with 10 more seconds of QPU
  time", "what is the smallest bond dimension you would trust and how do you know".
* Scoring sheet (judges): correctness of the error budget 4, ODR derivation 2, use of uncertainties
  (pulls, z-scores) 2, clarity 2.


===== CELL 109 [markdown] tags=[] =====
<span id="bonus"></span>
# Bonus (max +6, not part of the 100)

## B1 — dt → 0 at L = 34 by Richardson extrapolation (3 pts)

The hardware reference uses $dt=1$ (8 second-order steps). Build the $dt=0.5$ (16 steps) and
$dt=0.25$ (32 steps) physics circuits, count their two-qubit gates (this is what you would pay on
hardware), and — if you have the minutes — run them with Aer MPS at bond 32 and extrapolate with the
second-order Richardson formula

$$ \mathcal X^{(0)}_j \approx \frac{4\,\mathcal X^{(dt/2)}_j - \mathcal X^{(dt)}_j}{3}\,, $$

which cancels the leading $\mathcal O(dt^2)$ term of a second-order product formula. Compare with the
$dt=1$ profile: the difference is the Trotter contribution to your error budget at full scale.


===== CELL 110 [code] tags=[] =====
# PROMPT: build the dt = 0.5 and 0.25 physics circuits with evolve_circuits(..., n_steps=...), count 2-qubit gates, and fill
# bonus_B1 = {"n_steps": {dt: n}, "n2q": {dt: count}, "X": {dt: profile or None}, "X_richardson": array or None, "bond": 32};
# circuits_B1[dt] = (physics circuit, vacuum circuit).  The MPS part is guarded by RUN_BONUS_MPS (bond 32, several minutes
# per dt on a laptop); without it the grader scores the two circuits structurally and the Richardson part stays open.
RUN_BONUS_MPS = False


def count_2q(qc):
    """Number of two-qubit gates after expanding the RXX/RXY/barbell blocks into CX."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


def richardson(X_dt, X_half):
    """Second-order Richardson extrapolation from step sizes dt and dt/2."""
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER


bonus_B1 = {"n_steps": {}, "n2q": {}, "X": {}, "X_richardson": None, "bond": 32}
circuits_B1 = {}
for dt_ in (1.0, 0.5, 0.25):
    n_steps_ = int(round(T_FINAL / dt_))
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    bonus_B1["X"][dt_] = X_ref_bd64 if dt_ == 1.0 and X_ref_label == "MPS bond 64" else None
print("dt -> steps / 2q gates:", {k: (bonus_B1["n_steps"][k], bonus_B1["n2q"][k]) for k in bonus_B1["n2q"]})

_b1_ref = os.path.join("reference_data", "mps_reference_L34_dt.npz")      # organizer arrays (bond 32 for every dt), if released
if os.path.exists(_b1_ref):
    _d = np.load(_b1_ref)
    for dt_ in (1.0, 0.5, 0.25):
        if f"chi_wave_t8_dt{dt_}_bd32" in _d.files:
            bonus_B1["X"][dt_] = _d[f"chi_wave_t8_dt{dt_}_bd32"] - _d[f"chi_vacuum_t8_dt{dt_}_bd32"]
    print("loaded organizer bond-32 profiles for dt =", [k for k in (1.0, 0.5, 0.25) if bonus_B1["X"].get(k) is not None],
          "(all at the same bond dimension, so differences are pure Trotter error)")
elif RUN_BONUS_MPS:
    est_mps = AerEstimatorV2(options={"backend_options": {"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": 32,
                                                          "matrix_product_state_truncation_threshold": 1e-10}, "run_options": {"seed_simulator": 7}})
    for dt_ in (0.5, 0.25):
        t0_ = time.time()
        qc_dt, qv_dt = circuits_B1[dt_]
        r = est_mps.run([(qc_dt.decompose(reps=3), observables), (qv_dt.decompose(reps=3), observables)]).result()
        bonus_B1["X"][dt_] = np.asarray(r[0].data.evs) - np.asarray(r[1].data.evs)
        print(f"MPS bond 32, dt={dt_}: {time.time() - t0_:.0f} s")
if bonus_B1["X"].get(0.5) is not None and bonus_B1["X"].get(0.25) is not None:
    bonus_B1["X_richardson"] = richardson(bonus_B1["X"][0.5], bonus_B1["X"][0.25])
    print("Trotter budget at L=34, t=8:  max |X(dt=1) - X_Richardson| in W =", f"{np.max(np.abs((bonus_B1['X'][1.0] - bonus_B1['X_richardson'])[WINDOW])):.4f}",
          "| max |X(0.5) - X(0.25)| in W =", f"{np.max(np.abs((bonus_B1['X'][0.5] - bonus_B1['X'][0.25])[WINDOW])):.4f}",
          "| max |X(1) - X(0.5)| in W =", f"{np.max(np.abs((bonus_B1['X'][1.0] - bonus_B1['X'][0.5])[WINDOW])):.4f}")
    fig, ax = plt.subplots(figsize=(15, 3.5))
    for dt_, st in ((1.0, "g-o"), (0.5, "b--s"), (0.25, "m--^")):
        ax.plot(bonus_B1["X"][dt_], st, ms=4, label=f"dt = {dt_}")
    ax.plot(bonus_B1["X_richardson"], "k-", lw=2, label="Richardson dt -> 0")
    ax.set_xlim(WINDOW[0] - 3, WINDOW[-1] + 3); ax.set_xlabel("site j"); ax.set_ylabel(r"$\mathcal{X}_j$"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()
else:
    print("Richardson profile not computed (set RUN_BONUS_MPS = True or wait for the organizer arrays); circuits and gate counts are graded structurally.")


===== CELL 111 [code] tags=[] =====
_nan68 = np.full(2 * L, np.nan)
ff.grade_bonus_B1(circuits_B1[0.5][0], circuits_B1[0.25][0],
                  _nan68 if bonus_B1["X"].get(0.5) is None else bonus_B1["X"][0.5],
                  _nan68 if bonus_B1["X"].get(0.25) is None else bonus_B1["X"][0.25],
                  _nan68 if bonus_B1["X_richardson"] is None else bonus_B1["X_richardson"])


===== CELL 112 [markdown] tags=[] =====
## B2 — When does ODR fail? Amplitude damping vs depolarizing at L = 4 (2 pts)

Same four circuits as the L = 4 toy of 3.2 ($t=2$), simulated **exactly** (density matrix, no shot
noise) under (a) two-qubit depolarizing noise and (b) amplitude damping on every CZ
($p=\gamma=0.005$, suppression factors 0.7–0.9), each without and with your own Pauli twirls. To
isolate the effect of the *noise type* the calibration circuit must carry exactly the same gates as
the physics circuit, so here we transpile with `optimization_level=0` (no cancellation at the turning
point: 166 CZ in all four circuits). Report `toy_results = {"raw_depol", "odr_depol", "raw_amp",
"odr_amp", "odr_amp_twirl"}` as $\max_j|\mathcal X_j - \mathcal X^{\rm exact}_j|$, plus the charge witness
$\sum_j\langle Z_j\rangle$ of the physics circuit in each case.

What to expect, and to explain at the viva: with gate-matched circuits ODR removes most of the
depolarizing error ($\lesssim 25\,\%$ of the raw error survives — the rest is the state dependence of
the effective factors), amplitude damping is mitigated *worse* (its offset is state dependent), and
twirling the amplitude-damping circuits brings the error back towards the depolarizing value — but only
slowly (32 frames here; the residual witness tells you how far you are from a Pauli channel). The witness
ordering is unambiguous: *amplitude damping untwirled ≫ amplitude damping twirled > depolarizing*. With
the `optimization_level=1` circuits of 3.2 the picture changes: the physics/calibration mismatch of 3.1
then dominates all four cases (try it) — which is why Part 2.2 asked for a noise-matched calibration circuit.


===== CELL 113 [code] tags=[] =====
# PROMPT: fill toy_results = {"raw_depol", "odr_depol", "raw_amp", "odr_amp", "odr_amp_twirl"} (max |X - X_exact|) and
# bonus_B2 = {"rmse": {(noise, twirled): ...}, "maxabs": {...}, "witness": {...}, "ordering": [...]}; use AerEstimatorV2 with
# method="density_matrix" (precision 0 -> exact noisy expectation values) on the 8-qubit gate-matched ISA circuits
# (optimization_level=0, basis rz/sx/x/cz). Twirled runs average N_TWIRLS_B2[noise] twirl_circuit copies per circuit.
N_TWIRLS_B2, P_B2 = {"depolarizing": 8, "amplitude_damping": 32}, 0.005
qc_v4 = prep_wave(L4, TH_OV1, TH_OV3, EPS_VAC, EPS_VAC)
qc_v4_phys, qc_v4_mitig = evolve_circuits(qc_v4, L4, 2.0, m, g)
_, qc_w4_mitig = evolve_circuits(qc_w4, L4, 2.0, m, g)
pm_basis0 = generate_preset_pass_manager(optimization_level=0, basis_gates=["rz", "sx", "x", "cz"], seed_transpiler=1)
circuits_B2 = [pm_basis0.run(c) for c in (qc_w4_phys, qc_w4_mitig, qc_v4_phys, qc_v4_mitig)]
print("gate-matched circuits, CZ counts:", [c.count_ops().get("cz", 0) for c in circuits_B2])
obs4 = chiral_condensate_observables(L4)
X_exact_B2 = exact_chi(qc_w4_phys, L4) - exact_chi(qc_v4_phys, L4)
chi_w4_0, chi_v4_0 = exact_chi(qc_w4, L4), exact_chi(qc_v4, L4)
noise_dep_B2 = NoiseModel(); noise_dep_B2.add_all_qubit_quantum_error(depolarizing_error(P_B2, 2), "cz")
noise_ad_B2 = NoiseModel(); noise_ad_B2.add_all_qubit_quantum_error(amplitude_damping_error(P_B2).tensor(amplitude_damping_error(P_B2)), "cz")
noise_B2 = {"depolarizing": noise_dep_B2, "amplitude_damping": noise_ad_B2}
bonus_B2 = {"rmse": {}, "maxabs": {}, "maxabs_raw": {}, "witness": {}, "ordering": [], "p": P_B2, "n_twirls": N_TWIRLS_B2}
t0_ = time.time()
for nname, nm_ in noise_B2.items():
    est_dm = AerEstimatorV2(options={"backend_options": {"method": "density_matrix", "noise_model": nm_}})
    for twirled in (False, True):
        # BEGIN ANSWER
        # YOUR CODE HERE
        pass
        # END ANSWER
bonus_B2["ordering"] = sorted(bonus_B2["witness"], key=lambda k: -abs(bonus_B2["witness"][k]))
toy_results = {"raw_depol": bonus_B2["maxabs_raw"][("depolarizing", False)], "odr_depol": bonus_B2["maxabs"][("depolarizing", False)],
               "raw_amp": bonus_B2["maxabs_raw"][("amplitude_damping", False)], "odr_amp": bonus_B2["maxabs"][("amplitude_damping", False)],
               "odr_amp_twirl": bonus_B2["maxabs"][("amplitude_damping", True)], "odr_depol_twirl": bonus_B2["maxabs"][("depolarizing", True)]}
print(f"({time.time() - t0_:.0f} s)  case (noise, twirled): max|X - X_exact| raw -> ODR  (RMSE ODR) | witness sum<Z>")
for k in bonus_B2["rmse"]:
    print(f"   {str(k):34s}: {bonus_B2['maxabs_raw'][k]:.4f} -> {bonus_B2['maxabs'][k]:.4f}  ({bonus_B2['rmse'][k]:.4f}) | {bonus_B2['witness'][k]:+.4f}")
print("witness ordering (largest |sum<Z>| first):", bonus_B2["ordering"])
print("toy_results:", {k: round(v, 4) for k, v in toy_results.items()})


===== CELL 114 [code] tags=[] =====
ff.grade_bonus_B2(toy_results)


===== CELL 115 [markdown] tags=[] =====
## B3 — The barbell with fractional `rzz` gates (1 pt)

The barbell of `challenge_utils.barbell(a1..a6)` is a diagonal unitary: its 14 CX gates and 6
`rz` rotations implement $\exp\!\bigl(-\tfrac{i}{2}\sum_{j<k} a_{jk} Z_j Z_k\bigr)$ over four qubits
(no single-$Z$ terms). Heron devices expose the native **fractional** two-qubit gate
$R_{ZZ}(\theta) = \exp(-i\tfrac{\theta}{2} Z\otimes Z)$ for $0<\theta\le\pi/2$
(`service.backend(name, use_fractional_gates=True)`; angles outside that range are wrapped by the
`FoldRzzAngle` pass of `qiskit_ibm_runtime.transpiler.passes`, or by Qiskit's `WrapAngles` pass
when the target declares angle bounds). Rewrite the barbell with six `rzz` gates and verify the
unitary. Which pairs are nearest neighbours on the chain, and what does the transpiler have to do
about the other three?


===== CELL 116 [code] tags=[] =====
# PROMPT: barbell_rzz(a1, a2, a3, a4, a5, a6) -> 4-qubit QuantumCircuit made of rzz gates only that equals
# challenge_utils.barbell(a1..a6) up to a global phase. Hint: read the ZZ coefficients off the diagonal of Operator(barbell)
# (Walsh-Hadamard transform of the phases), then match each to an rzz angle.
def barbell_rzz(a1, a2, a3, a4, a5, a6):
    """Barbell as six RZZ rotations: angle a1 on (0,1), a2 on (1,2), a3 on (2,3), a4 on (0,2), a5 on (1,3), a6 on (0,3)."""
    qc = QuantumCircuit(4, name="barbell_rzz")
    # BEGIN ANSWER
    # YOUR CODE HERE
    # END ANSWER
    return qc


rng_b3 = np.random.default_rng(5)
for trial in range(5):
    angles = rng_b3.uniform(-1.5, 1.5, 6)
    assert Operator(barbell_rzz(*angles)).equiv(Operator(cu.barbell(*angles))), f"barbell_rzz differs from the CX barbell for {angles}"
print("barbell_rzz == challenge_utils.barbell (up to a global phase) on 5 random angle sets: OK")

from qiskit.transpiler import CouplingMap
angles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
line4 = CouplingMap.from_line(4)
pm_cz = generate_preset_pass_manager(optimization_level=3, basis_gates=["rz", "sx", "x", "cz"], coupling_map=line4, seed_transpiler=1)
pm_rzz = generate_preset_pass_manager(optimization_level=3, basis_gates=["rz", "sx", "x", "cz", "rzz"], coupling_map=line4, seed_transpiler=1)
qc_cx = QuantumCircuit(4); qc_cx.append(cu.barbell(*angles), range(4))
for label, pm_, circ in (("CX barbell -> cz basis", pm_cz, qc_cx), ("rzz barbell -> cz basis", pm_cz, barbell_rzz(*angles)),
                         ("rzz barbell -> cz+rzz basis (line)", pm_rzz, barbell_rzz(*angles))):
    isa_ = pm_.run(circ)
    n2 = {k: v for k, v in isa_.count_ops().items() if k in ("cz", "rzz")}
    print(f"{label:38s}: 2q gates {n2}, 2q depth {isa_.depth(lambda i: len(i.qubits) > 1)}")
print("Note: (0,2), (1,3), (0,3) are not chain neighbours, so the transpiler routes them with swaps (cz) or re-synthesises the "
      "diagonal; on hardware use use_fractional_gates=True and keep every rzz angle in (0, pi/2] (FoldRzzAngle / WrapAngles).")


===== CELL 117 [code] tags=[] =====
ff.grade_bonus_B3(barbell_rzz)


===== CELL 118 [markdown] tags=[] =====
<span id="closing"></span>
# Submission checklist

Your `submission/` folder must contain (the graders write `score.json` and the `ex*.npz` copies):

| file | produced by |
|---|---|
| `score.json`, `ex*.npz` | the `ff.grade_*` cells |
| `rehearsal_L6.npz` | 3.3 |
| `canary_t0.npz`, `canary_job_id.txt` (if run) | 4.1 |
| `hardware_t8.npz`, `job_id.txt` | 4.2 |
| `improvement.npz` | 4.3 |
| `error_budget.md`, `report.md` (+ figures) | Part 5 |
| this notebook, executed | you |

Zip the folder together with the executed notebook. Do **not** include your IBM Quantum token
anywhere. Run the summary below and check that every exercise you completed appears in it.


===== CELL 119 [code] tags=[] =====
ff.summary()
print("submission/ contains:", sorted(os.listdir("submission")))

