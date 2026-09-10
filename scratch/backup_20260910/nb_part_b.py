"""
Second half of the Fall Fest notebook (Parts 3-5, Bonus, closing) as a list of cells.

Cells are built with nbkit.md / nbkit.code; see organizer/SPEC.md section B for the
# PROMPT / # BEGIN ANSWER / # KEEP / # SOL conventions.  tools/build_notebooks.py
concatenates nb_part_a.CELLS + nb_part_b.CELLS.

Names assumed to exist from Parts 0-2 (SPEC section E):
  L, m, g, t, observables, qc_vacuum_init, qc_wave_init, chi_wave_exact, chi_vacuum_exact,
  qc_wave, qc_wave_mitig, qc_vacuum, qc_vacuum_mitig, backend, circuits_all_isa,
  observables_isa, qc_isa_init_layout, chain, flight_plan, estimator_options,
  prep_vacuum, prep_wave, trotter_step, evolve_circuits, chiral_condensate_observables
"""
from __future__ import annotations

from nbkit import md, code

CELLS: list[dict] = []
_add = CELLS.append

# =============================================================================
# Part 3 - Mitigation you wrote yourself
# =============================================================================
_add(md(r"""
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
"""))

_add(code(r'''
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
'''))

# ----------------------------------------------------------------------------- 3.1
_add(md(r"""
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
"""))

_add(code(r'''
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
    chi = np.asarray(chi, float)
    chi_cal = np.asarray(chi_cal, float)
    chi_exact = np.asarray(chi_exact, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1.0 - chi_cal) / (1.0 - chi_exact)                     # calibration factors, one per site
        fa, fb = f[..., :L], f[..., ::-1][..., :L]                   # site q and its mirror image 2L-1-q
        ca, cb = chi[..., :L], chi[..., ::-1][..., :L]
        keep_a = fa > suppression_threshold                          # post-selection masks
        keep_b = fb > suppression_threshold
        n_kept = keep_a.astype(float) + keep_b.astype(float)         # 0, 1 or 2 surviving members
        mean_chi = (np.where(keep_a, ca, 0.0) + np.where(keep_b, cb, 0.0)) / n_kept   # NaN where n_kept == 0
        mean_f = (np.where(keep_a, fa, 0.0) + np.where(keep_b, fb, 0.0)) / n_kept
        half = 1.0 - (1.0 - mean_chi) / mean_f
    return np.concatenate([half, half[..., ::-1]], axis=-1)
    # END ANSWER


def odr_uncertainty(chi, chi_std, chi_cal, chi_cal_std, chi_exact, L, suppression_threshold=0.01,
                    n_samples=2000, seed=0):
    """1-sigma uncertainty of odr_mitigate(...) by Monte-Carlo propagation of the Estimator stds."""
    # BEGIN ANSWER
    rng = np.random.default_rng(seed)
    chi = np.asarray(chi, float)
    chi_cal = np.asarray(chi_cal, float)
    s_phys = np.zeros_like(chi) if chi_std is None else np.asarray(chi_std, float)
    s_cal = np.zeros_like(chi_cal) if chi_cal_std is None else np.asarray(chi_cal_std, float)
    samples = rng.normal(chi, s_phys, size=(n_samples, chi.size))
    samples_cal = rng.normal(chi_cal, s_cal, size=(n_samples, chi_cal.size))
    mitigated = odr_mitigate(samples, samples_cal, chi_exact, L, suppression_threshold)   # (n_samples, 2L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # all-NaN columns are handled below
        sigma = np.nanstd(mitigated, axis=0, ddof=1)
    central = odr_mitigate(chi, chi_cal, chi_exact, L, suppression_threshold)
    sigma = np.where(np.isfinite(central), sigma, np.nan)
    return sigma
    # END ANSWER


def odr_bias(chi_true, f_phys, f_cal):
    """Systematic shift of the ODR estimate when the physics-circuit factor differs from the calibration factor."""
    # BEGIN ANSWER
    chi_true = np.asarray(chi_true, float)
    return (1.0 - chi_true) * (1.0 - np.asarray(f_phys, float) / np.asarray(f_cal, float))
    # END ANSWER
'''))

_add(md(r"""
**Self-test on synthetic data.** We fabricate a "true" $t=8$ profile (the symmetrised QDC bond-40
file), a set of site-dependent suppression factors including two hopeless sites (mirror pair $5
\leftrightarrow 62$ with $f=0.005$), and Gaussian shot noise. Three things must hold:
noise-free recovery is exact (except the two `NaN` sites), the bias formula reproduces the shift
observed when the physics factors are 10 % lower than the calibration factors, and the Monte-Carlo
$\sigma$ matches the scatter of repeated synthetic experiments.
"""))

_add(code(r'''
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
'''))

_add(code(r'''
# organizer sanity check: the hand-written ODR reproduces the QDC helper bit for bit
_r = np.random.default_rng(1)
for _ in range(20):
    a, b, c = _r.uniform(0, 2, 2 * L), _r.uniform(0, 2, 2 * L), _r.uniform(0.1, 1.9, 2 * L)
    mine, ref = odr_mitigate(a, b, c, L), cu.postselection_and_mitigation(a, b, c, L)
    assert np.array_equal(np.isnan(mine), np.isnan(ref)) and np.allclose(np.nan_to_num(mine), np.nan_to_num(ref), atol=1e-12)
print("odr_mitigate == challenge_utils.postselection_and_mitigation on 20 random inputs: OK")
''', tags=("solution-only",)))

_add(code(r'''
# grade 3.1 (synthetic data, fixed seeds)
ff.grade_ex3_1(odr_mitigate, odr_uncertainty, odr_bias)
'''))

# ----------------------------------------------------------------------------- 3.2
_add(md(r"""
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
"""))

_add(code(r'''
# PROMPT:
#   twirl_circuit(isa_circuit, seed) -> new circuit with random Pauli frames around EVERY cz, re-synthesised into
#       rz/sx/x so the circuit stays ISA (same qubits, same layout metadata: start from isa_circuit.copy_empty_like()).
#   postselect_charge(counts, L) -> the subset of `counts` whose bitstrings have Hamming weight L (charge Q_tot = 0).
#   z_from_counts(counts) -> array <Z_j> (j = qubit index, Qiskit little-endian bitstrings) and
#   charge_witness(counts) -> sum_j <Z_j>  (= -2 <Q_tot>).
# Hint: you may build the CZ conjugation table numerically with qiskit.quantum_info.Pauli/Operator instead of typing it.

# BEGIN ANSWER
_CZ_CONJ = {"II": "II", "IX": "ZX", "IY": "ZY", "IZ": "IZ",
            "XI": "XZ", "XX": "YY", "XY": "YX", "XZ": "XI",
            "YI": "YZ", "YX": "XY", "YY": "XX", "YZ": "YI",
            "ZI": "ZI", "ZX": "IX", "ZY": "IY", "ZZ": "ZZ"}     # key = Pauli on (q0, q1) before, value = after


def _apply_pauli_isa(qc, p, q):
    """Append the Pauli p on qubit q using ISA gates only (global phases are irrelevant)."""
    if p == "X":
        qc.x(q)
    elif p == "Z":
        qc.rz(np.pi, q)
    elif p == "Y":                       # Y ∝ Z·X : apply x first, then rz(pi)
        qc.x(q)
        qc.rz(np.pi, q)
# END ANSWER


def twirl_circuit(isa_circuit, seed):
    """Return an ISA-equivalent copy of `isa_circuit` with an independent random Pauli frame around every cz."""
    # BEGIN ANSWER
    rng = np.random.default_rng(seed)
    paulis = "IXYZ"
    out = isa_circuit.copy_empty_like()                          # keeps qubits, clbits and the transpiler layout
    for inst in isa_circuit.data:
        if inst.operation.name == "cz":
            q0, q1 = inst.qubits
            a, b = paulis[rng.integers(4)], paulis[rng.integers(4)]
            c, d = _CZ_CONJ[a + b]
            _apply_pauli_isa(out, a, q0); _apply_pauli_isa(out, b, q1)
            out.cz(q0, q1)
            _apply_pauli_isa(out, c, q0); _apply_pauli_isa(out, d, q1)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out
    # END ANSWER


def postselect_charge(counts, L):
    """Keep only the bitstrings with exactly L ones (total charge zero)."""
    # BEGIN ANSWER
    return {b: c for b, c in counts.items() if b.replace(" ", "").count("1") == L}
    # END ANSWER


def z_from_counts(counts):
    """<Z_j> for every qubit j from a counts dictionary (bitstring character n-1-j is qubit j)."""
    # BEGIN ANSWER
    keys = [b.replace(" ", "") for b in counts]
    n = len(keys[0])
    z = np.zeros(n)
    total = 0
    for b, c in zip(keys, counts.values()):
        total += c
        for j in range(n):
            z[j] += c * (1 - 2 * int(b[n - 1 - j]))
    return z / total
    # END ANSWER


def charge_witness(counts):
    """sum_j <Z_j> = -2 <Q_tot>; zero for any state in the Q_tot = 0 sector."""
    # BEGIN ANSWER
    return float(z_from_counts(counts).sum())
    # END ANSWER
'''))

_add(md(r"""
**Check and toy demonstration (L = 4, t = 2, Aer).** First the twirled circuit must be *exactly*
equivalent to the original (unitary check on an 8-qubit ISA circuit) and must contain only ISA gates.
Then we run the wavepacket physics circuit under two noise models of similar strength — two-qubit
depolarizing vs. amplitude damping on every CZ — and look at the charge witness and at the fraction of
shots that survive charge post-selection, with and without your twirls.
"""))

_add(code(r'''
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
'''))

_add(code(r'''
# grade 3.2 (unitary equivalence + ISA check of twirl_circuit on a hidden circuit; post-selection on hidden counts)
ff.grade_ex3_2(twirl_circuit, postselect_charge)
'''))

# ----------------------------------------------------------------------------- 3.3
_add(md(r"""
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
"""))

_add(code(r'''
# PROMPT: reduced_noise_model(backend, chain) -> qiskit_aer.noise.NoiseModel (see the text above), and
#         select_subchain(backend, chain, n_qubits=12) -> the contiguous window of `chain` with the smallest summed CZ error.
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


def _cz_error(target, a, b):
    """CZ error of edge (a, b) from the target, whichever direction is listed."""
    # BEGIN ANSWER
    props = target["cz"]
    key = (a, b) if (a, b) in props else (b, a)
    return float(props[key].error)
    # END ANSWER


def reduced_noise_model(backend, chain):
    """Hand-built Pauli noise model (depolarizing cz / sx / x + symmetric readout error) for the qubits in `chain`."""
    # BEGIN ANSWER
    target = backend.target
    chain = [int(q) for q in chain]
    nm = NoiseModel(basis_gates=["cz", "rz", "sx", "x"])
    for a, b in zip(chain[:-1], chain[1:]):
        eps_cz = _cz_error(target, a, b)
        for pair in ((a, b), (b, a)):
            nm.add_quantum_error(depolarizing_error(eps_cz, 2), "cz", list(pair))
    for q in chain:
        eps_sx = float(target["sx"][(q,)].error)
        nm.add_quantum_error(depolarizing_error(eps_sx, 1), ["sx", "x"], [q])
        p_ro = float(target["measure"][(q,)].error)
        nm.add_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]), [q])
    return nm
    # END ANSWER


def select_subchain(backend, chain, n_qubits=12):
    """Contiguous window of `chain` minimising the summed CZ error of its n_qubits - 1 edges."""
    # BEGIN ANSWER
    chain = [int(q) for q in chain]
    target = backend.target
    edge_err = [_cz_error(target, a, b) for a, b in zip(chain[:-1], chain[1:])]
    start = min(range(len(chain) - n_qubits + 1), key=lambda i: sum(edge_err[i:i + n_qubits - 1]))
    return chain[start:start + n_qubits]
    # END ANSWER


chain_L6 = select_subchain(backend, chain, 12)
noise_model = reduced_noise_model(backend, chain_L6)
print("12-qubit sub-chain:", chain_L6)
print("CZ errors on its edges:", [f"{_cz_error(backend.target, a, b):.4f}" for a, b in zip(chain_L6[:-1], chain_L6[1:])])
print("readout errors:", [f"{backend.target['measure'][(q,)].error:.3f}" for q in chain_L6])
print(noise_model)
'''))

_add(code(r'''
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
pm_L6 = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=chain_L6, seed_transpiler=42)
circuits_L6_isa = [pm_L6.run(c) for c in circuits_L6]
observables_L6_isa = [o.apply_layout(circuits_L6_isa[0].layout) for o in observables_L6]
estimator_L6 = AerEstimatorV2(options={
    "backend_options": {"method": "statevector", "noise_model": noise_model, "seed_simulator": 7},
    "run_options": {"shots": N_TRAJ_L6, "seed_simulator": 7},
})
result_L6 = estimator_L6.run([(c, observables_L6_isa) for c in circuits_L6_isa], precision=1 / np.sqrt(SHOTS_L6)).result()
names_L6 = ["chi_wave", "chi_wave_mitig", "chi_vacuum", "chi_vacuum_mitig"]
chi_raw_arrays = {k: np.asarray(r.data.evs, float) for k, r in zip(names_L6, result_L6)}
chi_std_arrays = {k: np.asarray(r.data.stds, float) for k, r in zip(names_L6, result_L6)}
# END ANSWER
print(f"rehearsal run: {time.time() - t0_:.1f} s")
for c in circuits_L6_isa:
    assert c.layout.initial_index_layout(filter_ancillas=True) == list(chain_L6), "all four circuits must sit on chain_L6"
    assert c.layout.final_index_layout() == list(chain_L6), "no SWAPs expected on a linear chain"
print("CZ counts:", [c.count_ops().get("cz", 0) for c in circuits_L6_isa],
      "| 2q depths:", [c.depth(lambda i: (not getattr(i.operation, "_directive", False)) and len(i.qubits) > 1) for c in circuits_L6_isa])
'''))

_add(code(r'''
# PROMPT: post-process the rehearsal with YOUR functions:
#   X_exact_L6 = exact chi(wave) - chi(vacuum) at t=4 (statevector);  calibration references = exact t=0 profiles
#   X_raw  = raw wave minus raw vacuum (site by site);  X_raw_cp = the same after CP (mirror) averaging of each arm
#   X_mit  = odr_mitigate(wave) - odr_mitigate(vacuum)            sigma_L6 = sqrt(sigma_wave^2 + sigma_vacuum^2) (odr_uncertainty)
#   rmse_raw_L6, rmse_mit_L6 vs X_exact_L6; pulls_L6 = (X_mit - X_exact)/sigma
# Names used below: chi_wave_exact_6, chi_vacuum_exact_6 (exact t=0 profiles), X_exact_L6, X_raw, X_raw_cp, X_mit, sigma_L6, pulls_L6.
# BEGIN ANSWER
chi_wave_exact_6 = exact_chi(qc_wave_init_6, L6)
chi_vacuum_exact_6 = exact_chi(qc_vacuum_init_6, L6)
X_exact_L6 = exact_chi(qc_wave_6, L6) - exact_chi(qc_vacuum_6, L6)
X_raw = chi_raw_arrays["chi_wave"] - chi_raw_arrays["chi_vacuum"]
X_raw_cp = cp_average(chi_raw_arrays["chi_wave"]) - cp_average(chi_raw_arrays["chi_vacuum"])
chi_wave_mit_6 = odr_mitigate(chi_raw_arrays["chi_wave"], chi_raw_arrays["chi_wave_mitig"], chi_wave_exact_6, L6)
chi_vacuum_mit_6 = odr_mitigate(chi_raw_arrays["chi_vacuum"], chi_raw_arrays["chi_vacuum_mitig"], chi_vacuum_exact_6, L6)
X_mit = chi_wave_mit_6 - chi_vacuum_mit_6
sig_w6 = odr_uncertainty(chi_raw_arrays["chi_wave"], chi_std_arrays["chi_wave"], chi_raw_arrays["chi_wave_mitig"], chi_std_arrays["chi_wave_mitig"], chi_wave_exact_6, L6)
sig_v6 = odr_uncertainty(chi_raw_arrays["chi_vacuum"], chi_std_arrays["chi_vacuum"], chi_raw_arrays["chi_vacuum_mitig"], chi_std_arrays["chi_vacuum_mitig"], chi_vacuum_exact_6, L6)
sigma_L6 = np.sqrt(sig_w6 ** 2 + sig_v6 ** 2)
rmse_raw_L6, rmse_mit_L6 = rmse(X_raw, X_exact_L6), rmse(X_mit, X_exact_L6)
pulls_L6 = (X_mit - X_exact_L6) / sigma_L6
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
'''))

_add(code(r'''
# grade 3.3 (noise model vs the frozen target summary, ODR consistency, RMSE vs the exact L=6 profile)
ff.grade_ex3_3(noise_model, chain_L6, chi_raw_arrays, X_raw, X_mit, odr_mitigate, backend)
'''))

# =============================================================================
# Part 4 - Hardware
# =============================================================================
_add(md(r"""
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
"""))

_add(code(r'''
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
'''))

# ----------------------------------------------------------------------------- 4.1
_add(md(r"""
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
"""))

_add(code(r'''
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
retention_wave = (1 - canary_evs[0]) / (1 - np.asarray(chi_wave_exact))
retention_vacuum = (1 - canary_evs[1]) / (1 - np.asarray(chi_vacuum_exact))
X0_exact = np.asarray(chi_wave_exact) - np.asarray(chi_vacuum_exact)
X0_meas = canary_evs[0] - canary_evs[1]
contrast_retention = float(np.max(X0_meas[[L - 1, L]]) / np.max(X0_exact[[L - 1, L]]))   # best of the two central sites
flagged_sites = [int(j) for j in np.where(retention_vacuum < 0.4)[0]]
flagged_sites_wave = [int(j) for j in np.where(retention_wave < 0.4)[0]]
flagged_qubits = [int(canary_layout[j]) for j in flagged_sites]
canary_pass = (len(flagged_sites) <= 2) and (contrast_retention >= 0.15)
verdict = "PASS" if canary_pass else "FAIL: re-select the chain (exclude the flagged qubits) before the main run"
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
''', tags=("hardware",)))

_add(code(r'''
# grade 4.1
ff.grade_ex4_1(canary_result, job_info_canary)
'''))

# ----------------------------------------------------------------------------- 4.2
_add(md(r"""
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
"""))

_add(code(r'''
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
    chi_w, chi_wc, chi_v, chi_vc = [np.asarray(e, float) for e in evs]
    std_w, std_wc, std_v, std_vc = [np.asarray(s, float) for s in stds]
    ce_w, ce_v = np.asarray(chi_wave_exact, float), np.asarray(chi_vacuum_exact, float)
    chi_mit_w = odr_mitigate(chi_w, chi_wc, ce_w, L, threshold)
    chi_mit_v = odr_mitigate(chi_v, chi_vc, ce_v, L, threshold)
    sig_w = odr_uncertainty(chi_w, std_w, chi_wc, std_wc, ce_w, L, threshold)
    sig_v = odr_uncertainty(chi_v, std_v, chi_vc, std_vc, ce_v, L, threshold)
    return {"X_raw": cp_average(chi_w) - cp_average(chi_v), "X_mit": chi_mit_w - chi_mit_v,
            "sigma": np.sqrt(sig_w ** 2 + sig_v ** 2), "chi_mit_wave": chi_mit_w, "chi_mit_vacuum": chi_mit_v,
            "factors_wave": (1 - chi_wc) / (1 - ce_w), "factors_vacuum": (1 - chi_vc) / (1 - ce_v),
            "sigma_raw": np.sqrt(cp_average(std_w ** 2) / 2 + cp_average(std_v ** 2) / 2)}
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
''', tags=("hardware",)))

_add(code(r'''
# grade 4.2 (recomputes X_hat from evs with YOUR odr_mitigate, then applies the leaderboard metric)
ff.grade_ex4_2(hardware_result, job_info, odr_mitigate)
'''))

# ----------------------------------------------------------------------------- 4.3
_add(md(r"""
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
"""))

_add(code(r'''
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
    xw, sw, rw = run["X_mit"][W], run["sigma"][W], X_ref_bd64[W]
    mask = np.isfinite(xw)
    n = int(mask.sum())
    val = float(np.sqrt(np.mean((xw[mask] - rw[mask]) ** 2)))
    grad = (xw[mask] - rw[mask]) / (n * val) if val > 0 else np.zeros(n)
    return val, float(np.sqrt(np.sum((grad * sw[mask]) ** 2)))
    # END ANSWER


def compare_runs(base, imp):
    """Headline z_score (RMSE_W change), z-scores per site and for the contrast; RMSE_W (+- sigma) of both runs."""
    # BEGIN ANSWER
    z_site = (imp["X_mit"] - base["X_mit"]) / np.sqrt(imp["sigma"] ** 2 + base["sigma"] ** 2)
    def contrast(run):
        X, s = run["X_mit"], run["sigma"]
        c = np.nanmean(X[[31, 32, 35, 36]]) - np.nanmean(X[[33, 34]])
        sc = np.sqrt(np.nansum(s[[31, 32, 35, 36]] ** 2) / 16 + np.nansum(s[[33, 34]] ** 2) / 4)
        return c, sc
    cb, sb = contrast(base); ci, si = contrast(imp)
    z_contrast = float((ci - cb) / np.sqrt(si ** 2 + sb ** 2))
    (rb, srb), (ri, sri) = rmse_W_with_sigma(base), rmse_W_with_sigma(imp)
    z_score = float((rb - ri) / np.sqrt(srb ** 2 + sri ** 2))
    return z_score, z_site, z_contrast, {"baseline": rb, "improved": ri, "sigma_baseline": srb, "sigma_improved": sri}
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
''', tags=("hardware",)))

_add(code(r'''
# grade 4.3
ff.grade_ex4_3(improvement, odr_mitigate)
'''))

# =============================================================================
# Part 5 - Error budget + defence
# =============================================================================
_add(md(r"""
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
"""))

_add(code(r'''
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
'''))

_add(md(r"""
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
"""))

# =============================================================================
# Bonus
# =============================================================================
_add(md(r"""
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
"""))

_add(code(r'''
# PROMPT: build the dt = 0.5 and 0.25 physics circuits with evolve_circuits(..., n_steps=...), count 2-qubit gates, and fill
# bonus_B1 = {"n_steps": {dt: n}, "n2q": {dt: count}, "X": {dt: profile or None}, "X_richardson": array or None, "bond": 32};
# circuits_B1[dt] = (physics circuit, vacuum circuit).  The MPS part is guarded by RUN_BONUS_MPS (bond 32, several minutes
# per dt on a laptop); without it the grader scores the two circuits structurally and the Richardson part stays open.
RUN_BONUS_MPS = False


def count_2q(qc):
    """Number of two-qubit gates after expanding the RXX/RXY/barbell blocks into CX."""
    # BEGIN ANSWER
    return sum(1 for inst in qc.decompose(reps=2).data if len(inst.qubits) == 2 and not getattr(inst.operation, "_directive", False))
    # END ANSWER


def richardson(X_dt, X_half):
    """Second-order Richardson extrapolation from step sizes dt and dt/2."""
    # BEGIN ANSWER
    return (4.0 * np.asarray(X_half, float) - np.asarray(X_dt, float)) / 3.0
    # END ANSWER


bonus_B1 = {"n_steps": {}, "n2q": {}, "X": {}, "X_richardson": None, "bond": 32}
circuits_B1 = {}
for dt_ in (1.0, 0.5, 0.25):
    n_steps_ = int(round(T_FINAL / dt_))
    # BEGIN ANSWER
    qc_dt, _ = evolve_circuits(qc_wave_init, L, T_FINAL, m, g, n_steps=n_steps_)
    qv_dt, _ = evolve_circuits(qc_vacuum_init, L, T_FINAL, m, g, n_steps=n_steps_)
    circuits_B1[dt_] = (qc_dt, qv_dt)
    bonus_B1["n_steps"][dt_] = n_steps_
    bonus_B1["n2q"][dt_] = count_2q(qc_dt)
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
'''))

_add(code(r'''
_nan68 = np.full(2 * L, np.nan)
ff.grade_bonus_B1(circuits_B1[0.5][0], circuits_B1[0.25][0],
                  _nan68 if bonus_B1["X"].get(0.5) is None else bonus_B1["X"][0.5],
                  _nan68 if bonus_B1["X"].get(0.25) is None else bonus_B1["X"][0.25],
                  _nan68 if bonus_B1["X_richardson"] is None else bonus_B1["X_richardson"])
'''))

_add(md(r"""
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
"""))

_add(code(r'''
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
        pass  # KEEP
        if twirled:
            n_tw = N_TWIRLS_B2[nname]
            pubs = [(twirl_circuit(c, seed=1000 * i + s_), obs4) for i, c in enumerate(circuits_B2) for s_ in range(n_tw)]
            evs_all = np.array([r.data.evs for r in est_dm.run(pubs, precision=0.0).result()]).reshape(4, n_tw, 2 * L4).mean(axis=1)
        else:
            evs_all = np.array([r.data.evs for r in est_dm.run([(c, obs4) for c in circuits_B2], precision=0.0).result()])
        X_raw_B2 = evs_all[0] - evs_all[2]
        X_mit_B2 = odr_mitigate(evs_all[0], evs_all[1], chi_w4_0, L4) - odr_mitigate(evs_all[2], evs_all[3], chi_v4_0, L4)
        z_phys = (-1.0) ** np.arange(2 * L4) * (evs_all[0] - 1)            # <Z_j> of the physics circuit from chi_j
        bonus_B2["rmse"][(nname, twirled)] = rmse(X_mit_B2, X_exact_B2)
        bonus_B2["maxabs"][(nname, twirled)] = float(np.nanmax(np.abs(X_mit_B2 - X_exact_B2)))
        bonus_B2["maxabs_raw"][(nname, twirled)] = float(np.max(np.abs(X_raw_B2 - X_exact_B2)))
        bonus_B2["witness"][(nname, twirled)] = float(z_phys.sum())
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
'''))

_add(code(r'''
ff.grade_bonus_B2(toy_results)
'''))

_add(md(r"""
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
"""))

_add(code(r'''
# PROMPT: barbell_rzz(a1, a2, a3, a4, a5, a6) -> 4-qubit QuantumCircuit made of rzz gates only that equals
# challenge_utils.barbell(a1..a6) up to a global phase. Hint: read the ZZ coefficients off the diagonal of Operator(barbell)
# (Walsh-Hadamard transform of the phases), then match each to an rzz angle.
def barbell_rzz(a1, a2, a3, a4, a5, a6):
    """Barbell as six RZZ rotations: angle a1 on (0,1), a2 on (1,2), a3 on (2,3), a4 on (0,2), a5 on (1,3), a6 on (0,3)."""
    qc = QuantumCircuit(4, name="barbell_rzz")
    # BEGIN ANSWER
    for theta, (j, k) in zip((a1, a2, a3, a4, a5, a6), ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3))):
        if abs(theta) > 0:
            qc.rzz(theta, j, k)
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
'''))

_add(code(r'''
ff.grade_bonus_B3(barbell_rzz)
'''))

# =============================================================================
# Closing
# =============================================================================
_add(md(r"""
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
"""))

_add(code(r'''
ff.summary()
print("submission/ contains:", sorted(os.listdir("submission")))
'''))
