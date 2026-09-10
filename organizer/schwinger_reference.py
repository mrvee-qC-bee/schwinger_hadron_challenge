"""
Reference (organizer-only) implementation for the Fall Fest edition of the
QDC 2025 "Hadron dynamics in the Schwinger model" challenge.

Everything here is verified against
  * the QDC MPS reference data (t=0 exact, t=8 bond-40) for L=34, and
  * exact linear algebra at small L (unitary identities, exact diagonalization).

Conventions (identical to the QDC notebook / arXiv:2401.08044):
  * 2L staggered sites -> 2L qubits, qubit j == staggered site j.
  * even sites = electrons, odd sites = positrons.
  * chiral condensate  chi_j = (-1)^j Z_j + I.
  * Qiskit little-endian Pauli strings: site j is character (n-1-j).

Do NOT ship this file to participants: it contains the full solutions.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import SparsePauliOp

# The participant-facing helper module (barbell, electric Trotter layer, RXXplus, ODR)
import challenge_utils as cu

# ---------------------------------------------------------------------------
# Model parameters (fixed by the pre-trained SC-ADAPT-VQE angles)
# ---------------------------------------------------------------------------
M_DEFAULT = 0.5
G_DEFAULT = 0.3
VACUUM_THETA_OV_1 = 0.30738
VACUUM_THETA_OV_3 = -0.04059
WAVE_THETA_O_11 = -1.6492
WAVE_THETA_O_22 = -0.3281


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def chiral_condensate_observables(L: int) -> list[SparsePauliOp]:
    """chi_j = (-1)^j Z_j + I for j = 0 .. 2L-1 (list of 2L SparsePauliOps)."""
    n = 2 * L
    obs = []
    for j in range(n):
        z = ["I"] * n
        z[n - 1 - j] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(z), (-1) ** j), ("I" * n, 1.0)]))
    return obs


def _pauli(n: int, ops: dict[int, str], coeff: float = 1.0) -> SparsePauliOp:
    s = ["I"] * n
    for j, p in ops.items():
        s[n - 1 - j] = p
    return SparsePauliOp.from_list([("".join(s), coeff)])


# ---------------------------------------------------------------------------
# Hamiltonians
# ---------------------------------------------------------------------------
def mass_hamiltonian(L: int, m: float = M_DEFAULT) -> SparsePauliOp:
    n = 2 * L
    H = 0 * _pauli(n, {})
    for j in range(n):
        H += _pauli(n, {j: "Z"}, m / 2 * (-1) ** j) + _pauli(n, {}, m / 2)
    return H.simplify()


def kinetic_hamiltonian(L: int) -> SparsePauliOp:
    """1/2 sum_j (sigma+_j sigma-_{j+1} + h.c.) = 1/4 sum_j (X_j X_{j+1} + Y_j Y_{j+1})."""
    n = 2 * L
    H = 0 * _pauli(n, {})
    for j in range(n - 1):
        H += _pauli(n, {j: "X", j + 1: "X"}, 0.25) + _pauli(n, {j: "Y", j + 1: "Y"}, 0.25)
    return H.simplify()


def electric_hamiltonian_full(L: int, g: float = G_DEFAULT) -> SparsePauliOp:
    """g^2/2 sum_{j=0}^{2L-2} (sum_{k<=j} Q_k)^2 with Q_k = -1/2 (Z_k + (-1)^k I).  Open boundaries."""
    n = 2 * L
    Q = [(_pauli(n, {k: "Z"}, -0.5) + _pauli(n, {}, -0.5 * (-1) ** k)) for k in range(n)]
    H = 0 * _pauli(n, {})
    cum = 0 * _pauli(n, {})
    for j in range(n - 1):
        cum = (cum + Q[j]).simplify()
        H += (g**2 / 2) * (cum @ cum)
    return H.simplify()


def electric_hamiltonian_truncated(L: int, g: float = G_DEFAULT) -> SparsePauliOp:
    """H_el^{(Q=0)}(1): the range-1 truncated, charge-zero-sector electric Hamiltonian
    used in the paper / QDC notebook (requires even L)."""
    assert L % 2 == 0, "the truncated electric Hamiltonian formula assumes even L"
    n = 2 * L
    half = L // 2
    T = []  # (dict qubit->Z, coefficient)   -- all Z-type
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
    H = 0 * _pauli(n, {})
    for ops, c in T:
        H += _pauli(n, ops, (g**2 / 2) * c)
    return H.simplify()


def schwinger_hamiltonian(L: int, m: float = M_DEFAULT, g: float = G_DEFAULT,
                          truncated: bool = False) -> SparsePauliOp:
    """Full Schwinger Hamiltonian H = H_m + H_kin + H_el (truncated=True uses H_el^{(Q=0)}(1))."""
    Hel = electric_hamiltonian_truncated(L, g) if truncated else electric_hamiltonian_full(L, g)
    return (mass_hamiltonian(L, m) + kinetic_hamiltonian(L) + Hel).simplify()


# ---------------------------------------------------------------------------
# Two-qubit building blocks (Fig. 5 of arXiv:2401.08044)
# ---------------------------------------------------------------------------
def RXYplus(theta: float) -> Gate:
    """R^{(XY)}_+(theta) = exp(-i theta/2 (XY + YX))  (verified numerically)."""
    qc = QuantumCircuit(2)
    qc.z(1); qc.h(1); qc.s(1)
    qc.s(0); qc.h(0)
    qc.cx(0, 1)
    qc.ry(theta, 0)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0); qc.sdg(0)
    qc.sdg(1); qc.h(1); qc.z(1)
    return qc.to_gate(label=rf"$R^{{XY}}_{{+}}({theta:.4g})$")


def RXYminus(theta: float) -> Gate:
    """R^{(XY)}_-(theta) = exp(+i theta/2 (XY - YX)) = exp(i theta O) with O = (XY - YX)/2  (verified numerically)."""
    qc = QuantumCircuit(2)
    qc.z(1); qc.h(1); qc.s(1)
    qc.s(0); qc.h(0)
    qc.cx(0, 1)
    qc.ry(-theta, 0)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0); qc.sdg(0)
    qc.sdg(1); qc.h(1); qc.z(1)
    return qc.to_gate(label=rf"$R^{{XY}}_{{-}}({theta:.4g})$")


RXXplus = cu.RXXplus  # R^{(XX)}_+(theta) = exp(-i theta/2 (XX + YY)) from challenge_utils (verified numerically)


# ---------------------------------------------------------------------------
# State preparation (SC-ADAPT-VQE, 2 steps each)
# ---------------------------------------------------------------------------
def prep_strong_coupling_vacuum(L: int) -> QuantumCircuit:
    """All sites empty: electrons (even sites) |1>, positrons (odd sites) |0>."""
    qc = QuantumCircuit(2 * L)
    for k in range(L):
        qc.x(2 * k)
    return qc


def vacuum_prep_rotate_OV_1(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    for k in range(L):
        qc.append(RXYminus(theta), [2 * k, 2 * k + 1])
    for k in range(L - 1):
        qc.append(RXYminus(-theta), [2 * k + 1, 2 * k + 2])
    return qc


def vacuum_prep_rotate_OV_3(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Fig. 4(b) of arXiv:2308.04481 (right-hand simplified form)."""
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
    return qc


def prep_vacuum(L: int, theta_OV_1: float = VACUUM_THETA_OV_1,
                theta_OV_3: float = VACUUM_THETA_OV_3) -> QuantumCircuit:
    qc = prep_strong_coupling_vacuum(L)
    qc = vacuum_prep_rotate_OV_1(qc, theta_OV_1, L)
    qc = vacuum_prep_rotate_OV_3(qc, theta_OV_3, L)
    return qc


def wave_prep_rotate_O_11(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    qc.append(RXYminus(theta), [L - 1, L])
    return qc


def wave_prep_rotate_O_22(qc: QuantumCircuit, theta: float, L: int) -> QuantumCircuit:
    """Fig. 6 (bottom) of arXiv:2401.08044, right-hand simplified form."""
    qc.append(RXYplus(-np.pi / 2), [L - 1, L])
    qc.append(RXYplus(-theta), [L - 2, L - 1])
    qc.append(RXYplus(-theta), [L, L + 1])
    qc.append(RXYplus(np.pi / 2), [L - 1, L])
    return qc


def prep_wave(L: int, theta_OV_1: float = VACUUM_THETA_OV_1, theta_OV_3: float = VACUUM_THETA_OV_3,
              theta_O_11: float = WAVE_THETA_O_11, theta_O_22: float = WAVE_THETA_O_22) -> QuantumCircuit:
    qc = prep_vacuum(L, theta_OV_1, theta_OV_3)
    qc = wave_prep_rotate_O_11(qc, theta_O_11, L)
    qc = wave_prep_rotate_O_22(qc, theta_O_22, L)
    return qc


def prep_vacuum_for_subtraction(L: int, eps: float = 0.9e-4) -> QuantumCircuit:
    """Vacuum circuit with the wavepacket layers applied at (almost) zero angle, so its
    structure/noise matches the wavepacket circuit (QDC convention)."""
    qc = prep_vacuum(L)
    qc = wave_prep_rotate_O_11(qc, eps, L)
    qc = wave_prep_rotate_O_22(qc, eps, L)
    return qc


# ---------------------------------------------------------------------------
# Trotter evolution (Fig. 8 of arXiv:2401.08044)
# ---------------------------------------------------------------------------
def trotter_step(qc: QuantumCircuit, L: int, time_step: float,
                 m: float = M_DEFAULT, g: float = G_DEFAULT) -> QuantumCircuit:
    """Second-order step: H_kin(t/2)  H_el(t)  H_m(t)  H_kin(t/2), gate order exactly as in
    Fig. 8 of arXiv:2401.08044: the first kinetic half-step applies the ODD bonds (1,2),(3,4),...
    first and then the even bonds (0,1),(2,3),...; the last half-step is the mirror image (even
    bonds first, then odd).  This is the ordering used by the QDC bond-40 reference files and by
    the cached ibm_kingston hardware run (verified 2026-09-07: bond-40 MPS at L=34, t=8 reproduces
    the QDC file to < 5e-5 with this ordering, vs 0.023 with even-first)."""
    n = 2 * L
    # H_kin over t/2 : odd bonds then even bonds, each R^{XX}_+(t/4)   (Fig. 8, first column)
    for j in range(1, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    for j in range(0, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    # H_el over t : single-qubit Z part (given in QDC) + 4-qubit barbell ZZ part
    for k in range(L // 2 - 1):
        qc.rz(g**2 * time_step, 2 * k)
        qc.rz(0.5 * g**2 * time_step, 2 * k + 1)
    qc.rz(0.5 * g**2 * time_step, L - 2)
    qc.rz(-0.5 * g**2 * time_step, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * time_step, L + 2 * k)
        qc.rz(-g**2 * time_step, L + 2 * k + 1)
    qc = cu.trotter_step_electric_2q(qc, L, time_step, g)
    # H_m over t
    for j in range(n):
        qc.rz((-1) ** j * m * time_step, j)
    # H_kin over t/2 (reversed sublattice order: even bonds then odd bonds, Fig. 8 last column)
    for j in range(0, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    for j in range(1, n - 1, 2):
        qc.append(RXXplus(time_step / 4), [j, j + 1])
    return qc


def evolve_circuits(qc_init: QuantumCircuit, L: int, t: float, m: float = M_DEFAULT,
                    g: float = G_DEFAULT, n_steps: int | None = None,
                    protect_midpoint: bool = False):
    """Physics circuit (n_steps forward) and ODR mitigation circuit (n_steps/2 forward,
    n_steps/2 backward).  n_steps defaults to the QDC choice 2*ceil(t/2) and must be even.
    protect_midpoint=True inserts a barrier at the turning point so the transpiler cannot
    cancel the last forward kinetic layer against the first backward one."""
    if n_steps is None:
        n_steps = int(2 * np.ceil(t / 2))
    assert n_steps % 2 == 0, "n_steps must be even for the forward/backward mitigation circuit"
    dt = t / n_steps
    qc = qc_init.copy()
    qc_mitig = qc_init.copy()
    for _ in range(n_steps):
        qc = trotter_step(qc, L, dt, m, g)
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, dt, m, g)
    if protect_midpoint:
        qc_mitig.barrier()
    for _ in range(n_steps // 2):
        qc_mitig = trotter_step(qc_mitig, L, -dt, m, g)
    return qc, qc_mitig


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def cp_symmetrize(chi: np.ndarray) -> np.ndarray:
    """CP symmetry <-> mirror symmetry j <-> 2L-1-j."""
    chi = np.asarray(chi, dtype=float)
    return 0.5 * (chi + chi[::-1])


def odr_mitigate(chi, chi_cal, chi_exact, L: int, suppression_threshold: float = 0.01) -> np.ndarray:
    """Operator decoherence renormalization with mirror-averaged suppression factors
    and post-selection (identical to QDC challenge_utils.postselection_and_mitigation)."""
    return cu.postselection_and_mitigation(np.asarray(chi), np.asarray(chi_cal), np.asarray(chi_exact),
                                           L, suppression_threshold=suppression_threshold)


def two_qubit_depth(qc: QuantumCircuit) -> int:
    return qc.depth(lambda i: (not getattr(i.operation, "_directive", False)) and len(i.qubits) > 1)


def signal_score(X_hw: np.ndarray, X_ref: np.ndarray) -> dict:
    """Leaderboard metrics for a vacuum-subtracted condensate profile X_j vs the MPS reference.
    * nrmse  : ||X_hw - X_ref||_2 / ||X_ref||_2 (NaNs in X_hw are treated as 0 -> penalized)
    * pearson: correlation coefficient over finite entries
    * score  : 100 * max(0, 1 - nrmse), i.e. 100 = perfect, 0 = as bad as reporting zeros
    """
    X_hw = np.asarray(X_hw, dtype=float)
    X_ref = np.asarray(X_ref, dtype=float)
    filled = np.where(np.isfinite(X_hw), X_hw, 0.0)
    nrmse = float(np.linalg.norm(filled - X_ref) / np.linalg.norm(X_ref))
    mask = np.isfinite(X_hw)
    pearson = float(np.corrcoef(X_hw[mask], X_ref[mask])[0, 1]) if mask.sum() > 2 else float("nan")
    return {"nrmse": nrmse, "pearson": pearson, "score": 100.0 * max(0.0, 1.0 - nrmse),
            "n_valid": int(mask.sum())}
