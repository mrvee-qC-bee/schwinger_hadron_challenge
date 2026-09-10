
# ===== cell 7 =====
import fallfest_grader as ff

ff.check_env()

# ===== cell 8 =====
import os
import time
import math
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# ===== cell 10 =====
# DO NOT MODIFY (the variational circuit parameters used in the rest of this challenge are specific to these model parameters)
m = 0.5   # electron/positron mass
g = 0.3   # coupling
L = 34    # number of spatial lattice sites, to be captured by 2 x L = 68 qubits

# ===== cell 13 =====
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

# ===== cell 14 =====
# grade your answer:
ff.grade_ex0_1(chiral_condensate_observables)

# ===== cell 16 =====
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

# ===== cell 18 =====
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

# ===== cell 19 =====
# grade your answer:
ff.grade_ex0_2(RXYplus, RXYminus)

# ===== cell 21 =====
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

# ===== cell 22 =====
# grade your answer (the grader calls prep_vacuum(L, 0.30738, -0.04059) at L = 6 and 8):
ff.grade_ex0_3(prep_vacuum)

# ===== cell 24 =====
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

# ===== cell 25 =====
# grade your answer (the grader calls prep_wave(L, th1, th3, th11, th22) at L = 6 and 8, with a random th22):
ff.grade_ex0_4(prep_wave)

# ===== cell 27 =====
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

# ===== cell 29 =====
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

# ===== cell 30 =====
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(range(2 * L), chi_wave_exact - chi_vacuum_exact, "ro", label="your circuits")
ax.plot(range(2 * L), chi_wave_exact_solution - chi_vacuum_exact_solution, "r--", linewidth=0.8, label="QDC reference")
ax.set_ylim(bottom=-0.25); ax.set_ylabel(r"$\mathcal{X}_j$"); ax.set_xlabel(r"Fermion staggered site $j$")
ax.set_title("Initial wavepacket, vacuum-subtracted (t = 0)"); ax.legend();

# ===== cell 32 =====
# PROMPT: Implement a full second-order Trotter step as in Fig. 8 (pinned ordering, see the text above).
# The electric part (Rz layer + barbells) is already filled in.
def trotter_step(qc: QuantumCircuit, L: int, time_step: float, m: float, g: float) -> QuantumCircuit:
    """Second-order step: H_kin(t/2)[odd, even]  H_el(t)[Rz layer, barbells]  H_m(t)  H_kin(t/2)[even, odd]  (Fig. 8)."""
    n = 2 * L
    # BEGIN ANSWER
    # YOUR CODE HERE
    # H_el over t: single-qubit Z part [ALREADY FILLED IN] + 4-qubit barbell ZZ part [given]
    for k in range(L // 2 - 1):
        qc.rz(g**2 * time_step, 2 * k)
        qc.rz(0.5 * g**2 * time_step, 2 * k + 1)
    qc.rz(0.5 * g**2 * time_step, L - 2)
    qc.rz(-0.5 * g**2 * time_step, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * time_step, L + 2 * k)
        qc.rz(-g**2 * time_step, L + 2 * k + 1)
    qc = trotter_step_electric_2q(qc, L, time_step, g)
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

# ===== cell 34 =====
qc_init6 = prep_wave(6, vacuum_prep_theta_OV_1, vacuum_prep_theta_OV_3, wave_prep_theta_O_11, wave_prep_theta_O_22)
qc6, qc6_mitig = evolve_circuits(qc_init6, 6, 4.0, m, g)
fid_return = abs(np.vdot(Statevector(qc_init6).data, Statevector(qc6_mitig).data)) ** 2
print(f"L=6, t=4: calibration circuit returns to the initial state with fidelity {fid_return:.12f}")

# ===== cell 35 =====
# grade your answer (the grader runs trotter_step at L = 6 with random dt (ratio test + pinned-ordering fidelity) and
# evolve_circuits(prep_wave(6, ...), 6, 2, m, g)):
ff.grade_ex0_5(trotter_step, evolve_circuits, prep_wave)

# ===== cell 37 =====
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

# ===== cell 40 =====
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

# ===== cell 41 =====
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

# ===== cell 42 =====
# grade your answer (Pauli dictionaries at L = 4, 6, 8, identity term excluded; E0_L8 to 1e-4):
ff.grade_ex1_1(schwinger_hamiltonian, electric_hamiltonian_truncated, E0_L8)

# ===== cell 44 =====
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

# ===== cell 45 =====
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

# ===== cell 46 =====
# grade your answer (electric_layer is compared with expm at L = 4 and 6 at a random t; truncation_shift to 1e-3):
ff.grade_ex1_2(electric_layer, truncation_shift)

# ===== cell 48 =====
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

# ===== cell 49 =====
# grade your answer:
ff.grade_ex1_3(vqe_fidelity, vqe_energy_gap)

# ===== cell 51 =====
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

# ===== cell 52 =====
# grade your answer (table entries to 5 %, ratios in [3, 5], Richardson below dt=0.25 error, dominant_error string):
ff.grade_ex1_4(trotter_table, richardson_error, dominant_error)

# ===== cell 54 =====
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

# ===== cell 55 =====
# grade your answer (mps_err must decrease with the bond dimension and match the organizer scan; X_bond40 within 3e-3 of the organizer bond 40):
ff.grade_ex1_5(mps_err, mps_seconds, X_bond40)

# ===== cell 58 =====
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

# ===== cell 59 =====
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

# ===== cell 60 =====
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

# ===== cell 61 =====
# grade your answer (ISA compliance vs backend.target, one common layout, 2q-depth range, observables mapped to the layout):
ff.grade_ex2_1(circuits_all_isa, observables_isa, backend)

# ===== cell 63 =====
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

# ===== cell 64 =====
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

# ===== cell 66 =====
qc_wave_mitig, qc_vacuum_mitig = qc_wave_mitig_matched, qc_vacuum_mitig_matched
circuits_all = [qc_wave, qc_wave_mitig, qc_vacuum, qc_vacuum_mitig]
circuits_all_isa = [pm_pinned.run(qc) for qc in circuits_all]
observables_isa = [obs.apply_layout(circuits_all_isa[0].layout) for obs in observables]
assert all(list(qc.layout.initial_index_layout(filter_ancillas=True)) == list(layout) for qc in circuits_all_isa)
print("CZ count per circuit:", [qc.count_ops().get("cz", 0) for qc in circuits_all_isa])

# ===== cell 67 =====
# grade your answer (counts vs the organizer's; evolve_circuits_matched is run at L = 6 and 8: return fidelity + CZ accounting on the given layout):
ff.grade_ex2_2(n2q_logical, n_cz_physics, evolve_circuits_matched, prep_wave, backend, layout)

# ===== cell 69 =====
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

# ===== cell 70 =====
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

# ===== cell 71 =====
# grade your answer (select_chain(backend) is called by the grader: valid 68-path, no dead edges/qubits, cost vs the frozen calibration):
ff.grade_ex2_3(select_chain, backend)

# ===== cell 73 =====
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

# ===== cell 75 =====
pubs = [(qc, obs_list) for qc, obs_list in zip(circuits_all_isa, [observables_isa] * 4)]
estimator = EstimatorV2(mode=backend, options=estimator_options)
print(f"{len(pubs)} PUBs x {len(observables_isa)} observables on {backend.name};",
      f"job would execute {flight_plan['main']['executions']} circuits (~{flight_plan['main']['usage_s']} s)")
print("Estimator options ready:", estimator.options.twirling.num_randomizations, "twirls x", estimator.options.twirling.shots_per_randomization, "shots")

# ===== cell 76 =====
# grade your answer (option fields, usage arithmetic, 4 ISA PUB circuits on one layout):
ff.grade_ex2_4(flight_plan, estimator_options, circuits_all_isa)
