"""
fallfest_grader.py -- LOCAL autograder for "Hadron Dynamics in the Schwinger Model"
(UofT Qiskit Fall Fest 2026 edition).  No network access, no IBM credentials.

    import fallfest_grader as ff
    ff.check_env()
    ff.grade_ex0_1(chiral_condensate_observables)
    ...
    ff.summary()

Every grade_ex* function
  1. validates the types of its inputs (raises TypeError/ValueError only on wrong types),
  2. runs deterministic checks (fixed seeds) against reference_data/grader_refs.npz,
  3. prints  "[ex1.2] 4.0/4 PASS -- ..."  or  "[ex1.2] 1.0/4 -- hint: ...",
  4. writes points + a compact copy of the inputs to submission/score.json and submission/exN.npz
     (organizer re-grading), and
  5. returns the points (float).  A wrong answer never raises.

Reference data are produced by organizer/make_grader_refs.py (the organizer module itself is
never imported here).  Tolerances and their calibration are documented in
organizer/BUILD_NOTES_grader.md.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import traceback
import warnings
from typing import Any, Callable

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(ROOT, "reference_data", "grader_refs.npz")
MPS_PATH = os.path.join(ROOT, "reference_data", "mps_reference_L34.npz")
SUBMISSION_DIR = os.environ.get("FF_SUBMISSION_DIR", os.path.join(ROOT, "submission"))

M_DEFAULT, G_DEFAULT = 0.5, 0.3
VACUUM_THETA_OV_1, VACUUM_THETA_OV_3 = 0.30738, -0.04059
WAVE_THETA_O_11, WAVE_THETA_O_22 = -1.6492, -0.3281

# usage model (SPEC E, QPU plan)
USAGE_OVERHEAD_S = 2.0
USAGE_PER_EXECUTION_S = 0.45e-3

MAX_POINTS = {
    "ex0.1": 2, "ex0.2": 2, "ex0.3": 2, "ex0.4": 2, "ex0.5": 2,
    "ex1.1": 5, "ex1.2": 4, "ex1.3": 3, "ex1.4": 5, "ex1.5": 3,
    "ex2.1": 3, "ex2.2": 5, "ex2.3": 5, "ex2.4": 2,
    "ex3.1": 7, "ex3.2": 5, "ex3.3": 8,
    "ex4.1": 3, "ex4.2": 18, "ex4.3": 4,
    "B1": 3, "B2": 2, "B3": 1,
}

_REFS: dict[str, np.ndarray] | None = None
_MPS: dict[str, np.ndarray] | None = None
_BASELINE_CACHE: dict[str, tuple[float, list[int]]] = {}


# ---------------------------------------------------------------------------
# infrastructure
# ---------------------------------------------------------------------------
def _refs() -> dict[str, np.ndarray]:
    global _REFS
    if _REFS is None:
        if not os.path.exists(REF_PATH):
            raise FileNotFoundError(f"{REF_PATH} missing -- run organizer/make_grader_refs.py")
        with np.load(REF_PATH, allow_pickle=False) as d:
            _REFS = {k: d[k] for k in d.files}
    return _REFS


def _mps() -> dict[str, np.ndarray]:
    global _MPS
    if _MPS is None:
        if not os.path.exists(MPS_PATH):
            raise FileNotFoundError(f"{MPS_PATH} missing -- run tools/make_mps_reference.py")
        with np.load(MPS_PATH, allow_pickle=False) as d:
            _MPS = {k: d[k] for k in d.files}
    return _MPS


def x_ref_t8(bond: int = 64) -> np.ndarray:
    """Organizer MPS reference X_ref = chi_wave(t=8) - chi_vacuum(t=8) at L=34."""
    d = _mps()
    return d[f"chi_wave_t8_bd{bond}"] - d[f"chi_vacuum_t8_bd{bond}"]


def _score_path() -> str:
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    return os.path.join(SUBMISSION_DIR, "score.json")


def _load_scores() -> dict:
    p = _score_path()
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _record(ex: str, points: float, detail: str, arrays: dict | None = None,
            extra: dict | None = None) -> float:
    maxp = MAX_POINTS[ex]
    points = float(min(max(points, 0.0), maxp))
    verdict = "PASS" if points >= maxp - 1e-9 else ("FAIL" if points <= 0 else "PARTIAL")
    tag = "" if verdict == "PASS" else " -- hint:"
    print(f"[{ex}] {points:.1f}/{maxp} {verdict if verdict == 'PASS' else ''}{'' if verdict == 'PASS' else tag} {detail}".replace("  ", " "))
    scores = _load_scores()
    entry = {"points": points, "max": maxp, "detail": detail, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        entry.update(_jsonable(extra))
    scores[ex] = entry
    with open(_score_path(), "w") as f:
        json.dump(scores, f, indent=1)
    if arrays:
        clean = {}
        for k, v in arrays.items():
            try:
                a = np.asarray(v)
                if a.dtype == object:
                    a = np.asarray(json.dumps(_jsonable(v)))
                clean[k] = a
            except Exception:
                clean[k] = np.asarray(str(v))
        np.savez(os.path.join(SUBMISSION_DIR, f"{ex.replace('.', '_')}.npz"), **clean)
    return points


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _call(fn: Callable, *args, **kw):
    """Call a participant function; return (result, error_string)."""
    try:
        return fn(*args, **kw), None
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        return None, f"{type(e).__name__}: {e} ({tb[:120]})"


def _require_callable(name: str, fn: Any) -> None:
    if not callable(fn):
        raise TypeError(f"{name} must be a function (got {type(fn).__name__})")


def _require_type(name: str, obj: Any, types) -> None:
    if not isinstance(obj, types):
        tn = types.__name__ if isinstance(types, type) else "/".join(t.__name__ for t in types)
        raise TypeError(f"{name} must be {tn} (got {type(obj).__name__})")


def _qc_types():
    from qiskit import QuantumCircuit
    return QuantumCircuit


def _statevector(qc) -> np.ndarray:
    from qiskit.quantum_info import Statevector
    return Statevector(qc).data


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.shape != b.shape:
        return 0.0
    return float(abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def _phase_aligned_maxdiff(u: np.ndarray, v: np.ndarray) -> float:
    """max |u - e^{i phi} v| with the global phase chosen from the overlap."""
    ov = np.vdot(v, u)
    ph = ov / abs(ov) if abs(ov) > 1e-300 else 1.0
    return float(np.max(np.abs(u - ph * v)))


def _operator_close(U, V, atol=1e-8) -> tuple[bool, float]:
    U = np.asarray(U); V = np.asarray(V)
    if U.shape != V.shape:
        return False, float("inf")
    i = np.unravel_index(np.argmax(np.abs(V)), V.shape)
    ph = U[i] / V[i]
    err = float(np.max(np.abs(U - ph * V)))
    return err < atol, err


def _pauli_dict(op, drop_identity: bool = True) -> dict[str, complex]:
    """SparsePauliOp -> {label: coeff} (simplified, identity excluded).  Uses Pauli.to_label(): str(Pauli)
    truncates labels longer than 50 qubits, which made 156-qubit observables collide."""
    from qiskit.quantum_info import SparsePauliOp
    if not isinstance(op, SparsePauliOp):
        raise TypeError(f"expected SparsePauliOp, got {type(op).__name__}")
    op = op.simplify(atol=1e-12)
    out = {}
    for p, c in zip(op.paulis, op.coeffs):
        lab = p.to_label()
        if drop_identity and set(lab) == {"I"}:
            continue
        out[lab] = complex(c)
    return out


# ---------------------------------------------------------------------------
# Reference fingerprints (the shipped grader_refs.npz holds NO answers in clear: scalars and Pauli
# dictionaries are compared through quantised SHA-256 hashes, statevectors through random projections,
# unitaries through their action on stored seeded random states).  organizer/make_grader_refs.py uses
# the same helpers, so the two sides stay consistent.
# ---------------------------------------------------------------------------
HASH_GRIDS = (0.0, 0.5)          # two quantisation grids offset by half a cell: a value within q/2 of the
                                 # reference lands in the same cell on at least one of them
FINGERPRINT_K = 24               # random projections per statevector fingerprint


def _quantised_bytes(values, q: float, offset: float) -> bytes:
    v = np.asarray(values, dtype=float).ravel()
    cells = np.floor(v / q + offset).astype(np.int64)
    return cells.tobytes()


def value_hashes(values, q: float) -> list[str]:
    """SHA-256 of the quantised value(s) on the two grids."""
    import hashlib
    return [hashlib.sha256(_quantised_bytes(values, q, off)).hexdigest() for off in HASH_GRIDS]


def pauli_dict_hashes(d: dict[str, complex], q: float = 1e-6) -> list[str]:
    """Hash of a Pauli dictionary: sorted labels + quantised real/imag coefficients (two grids)."""
    import hashlib
    out = []
    for off in HASH_GRIDS:
        h = hashlib.sha256()
        for lab in sorted(d):
            c = complex(d[lab])
            h.update(lab.encode()); h.update(_quantised_bytes([c.real, c.imag], q, off))
        out.append(h.hexdigest())
    return out


def _hash_match(hashes, ref_hashes) -> bool:
    ref = [str(x) for x in np.asarray(ref_hashes).ravel()]
    return any(h == rh for h, rh in zip(hashes, ref))


def _value_match(value, ref_key: str, q: float) -> bool:
    return _hash_match(value_hashes(value, q), _refs()[ref_key])


def fingerprint_basis(dim: int, K: int = FINGERPRINT_K, seed: int = 7001) -> np.ndarray:
    """K seeded random complex vectors (rows) of length dim; not normalised (|<phi|psi>| ~ 1)."""
    rng = np.random.default_rng(seed + dim)
    return rng.normal(size=(K, dim)) + 1j * rng.normal(size=(K, dim))


def fingerprint(psi: np.ndarray) -> np.ndarray:
    """Random-projection fingerprint c_k = <phi_k|psi> of a statevector (K complex numbers)."""
    psi = np.asarray(psi).ravel()
    return fingerprint_basis(psi.size) @ psi


def _fingerprint_distance(psi: np.ndarray, ref_fp: np.ndarray) -> float:
    """Phase-aligned max |c_k(psi) - e^{i phi} c_k(ref)| / rms |c_k(ref)|; ~1e-13 for identical states,
    O(fidelity loss) otherwise.  Returns inf for a dimension mismatch."""
    psi = np.asarray(psi).ravel()
    ref_fp = np.asarray(ref_fp).ravel()
    if psi.size != 2 ** int(round(math.log2(psi.size))) or ref_fp.size != FINGERPRINT_K:
        return float("inf")
    c = fingerprint(psi / np.linalg.norm(psi))
    ov = np.vdot(ref_fp, c)
    ph = ov / abs(ov) if abs(ov) > 1e-300 else 1.0
    return float(np.max(np.abs(c - ph * ref_fp)) / np.sqrt(np.mean(np.abs(ref_fp) ** 2)))


def _fp_match(psi, ref_key: str, tol: float = 1e-6) -> tuple[bool, float]:
    r = _refs()
    if ref_key not in r:
        return False, float("inf")
    ref = r[ref_key]
    if np.asarray(psi).size != int(r.get(ref_key + "_dim", np.asarray(psi).size)):
        return False, float("inf")
    d = _fingerprint_distance(psi, ref)
    return d < tol, d


def _as_number(x):
    """Coerce a scalar-like (int/float/numpy scalar/0-d or size-1 array/numeric string) to float; None otherwise."""
    if isinstance(x, bool):
        return None
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    try:
        a = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if a.size != 1:
        return None
    return float(a[0])


def _as_int(x):
    v = _as_number(x)
    if v is None or not np.isfinite(v) or abs(v - round(v)) > 1e-9:
        return None
    return int(round(v))


def _strip_measurements(qc):
    """Circuit without final measurements / classical registers (participants sometimes measure_all)."""
    if qc.num_clbits == 0:
        return qc
    try:
        return qc.remove_final_measurements(inplace=False)
    except Exception:  # noqa: BLE001
        return qc


def _dict_maxdiff(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return max((abs(a.get(k, 0) - b.get(k, 0)) for k in keys), default=0.0)


def _chi_observables(L: int):
    from qiskit.quantum_info import SparsePauliOp
    n = 2 * L
    obs = []
    for j in range(n):
        z = ["I"] * n; z[n - 1 - j] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(z), (-1) ** j), ("I" * n, 1.0)]))
    return obs


def _chi_from_state(psi: np.ndarray, L: int) -> np.ndarray:
    from qiskit.quantum_info import Statevector
    sv = Statevector(psi)
    return np.array([float(np.real(sv.expectation_value(o))) for o in _chi_observables(L)])


def _qdc_odr(chi, chi_cal, chi_exact, L, suppression_threshold=0.01) -> np.ndarray:
    """Grader-internal copy of the QDC pooled ODR formula (mirror-averaged suppression factors)."""
    chi = np.asarray(chi, float); chi_cal = np.asarray(chi_cal, float); chi_exact = np.asarray(chi_exact, float)
    out = np.zeros(2 * L)
    f = (1 - chi_cal) / (1 - chi_exact)
    for q in range(L):
        sel_c, sel_f = [], []
        for s in (q, 2 * L - 1 - q):
            if f[s] > suppression_threshold:
                sel_c.append(chi[s]); sel_f.append(f[s])
        out[q] = 1 - (1 - np.mean(sel_c)) / np.mean(sel_f) if sel_c else np.nan
    out[2 * L - 1:L - 1:-1] = out[:L]
    return out


def _two_qubit_depth(qc) -> int:
    return qc.depth(lambda i: (not getattr(i.operation, "_directive", False)) and len(i.qubits) > 1)


def _n2q(qc) -> int:
    return sum(1 for i in qc.decompose(reps=4).data if len(i.qubits) > 1 and not getattr(i.operation, "_directive", False))


def _run_with_timeout(fn: Callable, timeout_s: float, *args, **kw):
    """Run fn in a daemon thread; returns (result, error, elapsed, timed_out)."""
    box: dict[str, Any] = {}

    def _target():
        try:
            box["res"] = fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            box["err"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=_target, daemon=True)
    t0 = time.time(); th.start(); th.join(timeout_s); el = time.time() - t0
    if th.is_alive():
        return None, None, el, True
    return box.get("res"), box.get("err"), el, False


# ---------------------------------------------------------------------------
# backend / target helpers
# ---------------------------------------------------------------------------
def target_summary(backend) -> dict[str, np.ndarray]:
    """Frozen summary for the three Open-Plan fake backends (from grader_refs.npz), live otherwise."""
    name = getattr(backend, "name", "") or ""
    key = None
    for k in ("kingston", "fez", "marrakesh"):
        if k in name.lower():
            key = k
    r = _refs()
    if key is not None and f"tgt_{key}_edges" in r:
        s = {k: r[f"tgt_{key}_{k}"] for k in ("num_qubits", "edges", "cz_error", "sx_error", "x_error", "readout_error", "t1", "t2")}
        s["name"] = f"frozen:{key}"
        return s
    return live_target_summary(backend)


def live_target_summary(backend) -> dict[str, np.ndarray]:
    tgt = backend.target
    nq = tgt.num_qubits
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    twoq = "cz" if "cz" in tgt.operation_names else next(n for n in tgt.operation_names if n in ("ecr", "cx", "rzz"))
    cz = []
    for (a, b) in edges:
        p = tgt[twoq].get((a, b)) or tgt[twoq].get((b, a))
        cz.append(np.nan if p is None or p.error is None else float(p.error))

    def q1(name):
        out = []
        for q in range(nq):
            p = tgt[name].get((q,)) if name in tgt.operation_names else None
            out.append(np.nan if p is None or p.error is None else float(p.error))
        return np.array(out)

    t1 = []; t2 = []
    for q in range(nq):
        qp = tgt.qubit_properties[q] if tgt.qubit_properties is not None else None
        t1.append(np.nan if qp is None or qp.t1 is None else float(qp.t1))
        t2.append(np.nan if qp is None or qp.t2 is None else float(qp.t2))
    return {"name": f"live:{getattr(backend, 'name', '?')}", "num_qubits": np.asarray(nq), "edges": np.array(edges, dtype=int),
            "cz_error": np.array(cz), "sx_error": q1("sx"), "x_error": q1("x"), "readout_error": q1("measure"),
            "t1": np.array(t1), "t2": np.array(t2)}


# Chain cost = expected number of errors on the t=8 physics circuit (log form), as defined in the notebook
# (ex 2.3): ~74 CZ per bond (4958 CZ / 67 bonds on FakeKingston) and ~150 sx per qubit, readout once per qubit.
N_CZ_PER_BOND = 74.0
N_SX_PER_QUBIT = 150.0


def _cost_tables(s: dict):
    nq = int(s["num_qubits"]); ro = np.asarray(s["readout_error"], float)
    sx = np.asarray(s.get("sx_error", np.full(nq, np.nan)), float)
    sx_cost = np.where(np.isfinite(sx), -np.log1p(-np.clip(np.nan_to_num(sx), 0.0, 0.999)), 0.0)   # unknown sx error -> 0
    node = np.full(nq, np.inf)
    ok = np.isfinite(ro) & (ro < 0.5)
    node[ok] = -np.log1p(-ro[ok]) + N_SX_PER_QUBIT * sx_cost[ok]
    adj: dict[int, dict[int, float]] = {q: {} for q in range(nq)}
    for (a, b), e in zip(s["edges"], s["cz_error"]):
        if np.isfinite(e) and e < 0.5:
            c = N_CZ_PER_BOND * (-math.log1p(-float(e))); adj[int(a)][int(b)] = c; adj[int(b)][int(a)] = c
    return node, adj


def chain_cost(summary: dict, chain) -> float:
    """C = sum_bonds 74 (-ln(1-eps_CZ)) + sum_qubits [150 (-ln(1-eps_sx)) + (-ln(1-eps_RO))]  (expected errors per
    circuit, the ex 2.3 objective); inf for invalid chains."""
    node, adj = _cost_tables(summary)
    chain = [int(q) for q in chain]
    c = 0.0
    for q in chain:
        if q < 0 or q >= len(node) or not np.isfinite(node[q]):
            return float("inf")
        c += node[q]
    for a, b in zip(chain[:-1], chain[1:]):
        if b not in adj[a]:
            return float("inf")
        c += adj[a][b]
    return float(c)


def chain_validity(summary: dict, chain, n_qubits: int = 68) -> tuple[bool, str]:
    try:
        chain = [int(q) for q in chain]
    except Exception:
        return False, "chain must be a list of ints"
    if len(chain) != n_qubits:
        return False, f"chain has {len(chain)} qubits, expected {n_qubits}"
    if len(set(chain)) != n_qubits:
        return False, "chain has repeated qubits"
    nq = int(summary["num_qubits"])
    if min(chain) < 0 or max(chain) >= nq:
        return False, "chain contains qubit indices outside the device"
    node, adj = _cost_tables(summary)
    for a, b in zip(chain[:-1], chain[1:]):
        if b not in adj[a]:
            e = _edge_error(summary, a, b)
            if e is None:
                return False, f"({a},{b}) is not an edge of the coupling map"
            return False, f"edge ({a},{b}) has CZ error {e:.3f} >= 0.5 (dead edge)"
    for q in chain:
        if not np.isfinite(node[q]):
            return False, f"qubit {q} has readout error >= 0.5 or unknown"
    return True, "ok"


def _edge_error(summary, a, b):
    for (x, y), e in zip(summary["edges"], summary["cz_error"]):
        if {int(x), int(y)} == {int(a), int(b)}:
            return float(e)
    return None


def baseline_chain(summary: dict, n_qubits: int = 68, seed: int = 2026, n_restarts: int = 2000,
                   budget_s: float = 10.0, temperature: float = 1.0) -> tuple[float, list[int]]:
    """Grader baseline: seeded randomized greedy DFS (cost-sorted neighbours + Gumbel noise), fixed
    number of restarts (deterministic), 10 s safety budget.  Returns (cost, chain)."""
    node, adj = _cost_tables(summary)
    nq = len(node)
    rng = np.random.default_rng(seed)
    good = [q for q in range(nq) if np.isfinite(node[q]) and adj[q]]
    # noise scale ~ the typical (bond + qubit) cost, so the randomisation is independent of the cost units
    noise = temperature * (float(np.median([c for q in adj for c in adj[q].values()] or [0.0])) + float(np.median(node[good])))
    best_cost, best = float("inf"), []
    t0 = time.time()
    for _ in range(n_restarts):
        if time.time() - t0 > budget_s:
            break
        start = int(rng.choice(good)); path = [start]; visited = {start}

        def cands(q):
            cs = [(adj[q][v] + node[v] + noise * rng.gumbel(), v) for v in adj[q] if v not in visited]
            cs.sort()
            return [v for _, v in cs]

        stack = [cands(start)]; expansions = 0
        while stack and len(path) < n_qubits and expansions < 20000:
            if stack[-1]:
                v = stack[-1].pop(0); path.append(v); visited.add(v); stack.append(cands(v)); expansions += 1
            else:
                stack.pop(); visited.discard(path.pop())
        if len(path) == n_qubits:
            c = chain_cost(summary, path)
            if c < best_cost:
                best_cost, best = c, list(path)
    return best_cost, best


def _baseline_for(summary: dict, n_qubits: int = 68) -> tuple[float, list[int]]:
    key = f"{summary.get('name', '?')}:{n_qubits}"
    if key not in _BASELINE_CACHE:
        _BASELINE_CACHE[key] = baseline_chain(summary, n_qubits)
    return _BASELINE_CACHE[key]


def _is_isa(qc, backend) -> tuple[bool, str]:
    tgt = backend.target
    for inst in qc.data:
        name = inst.operation.name
        if name == "barrier" or getattr(inst.operation, "_directive", False):
            continue
        qargs = tuple(qc.find_bit(q).index for q in inst.qubits)
        if not tgt.instruction_supported(operation_name=name, qargs=qargs):
            return False, f"'{name}' on {qargs} not supported by {backend.name}.target"
    return True, "ok"


def _layouts(qc):
    if qc.layout is None:
        return None, None
    return list(qc.layout.initial_index_layout(filter_ancillas=True)), list(qc.layout.final_index_layout())


def _cz_pairs(qc) -> list[tuple[int, int]]:
    out = []
    for inst in qc.data:
        if len(inst.qubits) == 2 and not getattr(inst.operation, "_directive", False):
            out.append(tuple(qc.find_bit(q).index for q in inst.qubits))
    return out


def ex05_random_states(n: int = 12, k: int = 3, seed: int = 1234) -> list[np.ndarray]:
    """The seeded random states of the ex 0.5 ratio test (shared with organizer/make_grader_refs.py)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n); out.append(v / np.linalg.norm(v))
    return out


EX05_DTS = (0.4, 0.2, 0.1)


def ex12_random_inputs(seed: int = 1202) -> dict[int, tuple[float, list[np.ndarray]]]:
    """{L: (t, [3 random states])} of the ex 1.2 electric-layer test (shared with make_grader_refs.py)."""
    rng = np.random.default_rng(seed)
    out = {}
    for L in (4, 6):
        n = 2 * L; t = float(rng.uniform(0.15, 0.6)); vs = []
        for _ in range(3):
            v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n); vs.append(v / np.linalg.norm(v))
        out[L] = (t, vs)
    return out


def ex32_base_circuit():
    """Grader-owned ISA circuit for the twirling test: a seeded random rz/sx/cz brickwork on 12 qubits transpiled
    (O1, seed 42) onto the frozen 12-qubit FakeKingston chain stored in grader_refs.npz.  Returns (circuit, active
    physical qubits).  Self-referential, so it does not depend on the transpiler version."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    r = _refs()
    active = [int(q) for q in r["ex32_layout"]]
    rng = np.random.default_rng(3200)
    qc = QuantumCircuit(12)
    for layer in range(44):
        for q in range(12):
            qc.rz(float(rng.uniform(-np.pi, np.pi)), q); qc.sx(q); qc.rz(float(rng.uniform(-np.pi, np.pi)), q)
        for j in range(layer % 2, 11, 2):
            qc.cz(j, j + 1)
    pm = generate_preset_pass_manager(optimization_level=1, backend=FakeKingston(), initial_layout=active, seed_transpiler=42)
    return pm.run(qc), active


# ---------------------------------------------------------------------------
# env / summary
# ---------------------------------------------------------------------------
def check_env() -> dict:
    """Print package versions; warn if below the tested versions."""
    import qiskit
    info = {"qiskit": qiskit.__version__}
    try:
        import qiskit_ibm_runtime
        info["qiskit_ibm_runtime"] = qiskit_ibm_runtime.__version__
    except Exception:
        info["qiskit_ibm_runtime"] = None
    try:
        import qiskit_aer
        info["qiskit_aer"] = qiskit_aer.__version__
    except Exception:
        info["qiskit_aer"] = None
    import scipy
    info["numpy"] = np.__version__; info["scipy"] = scipy.__version__

    def _ge(v, minimum):
        try:
            return tuple(int(x) for x in v.split(".")[:2]) >= minimum
        except Exception:
            return False

    ok = True
    for k, v in info.items():
        print(f"  {k:20s} {v}")
    if not _ge(info["qiskit"], (2, 5)):
        warnings.warn("qiskit >= 2.5 is required (pip install -r requirements.txt)"); ok = False
    if info["qiskit_ibm_runtime"] is None or not _ge(info["qiskit_ibm_runtime"], (0, 49)):
        warnings.warn("qiskit-ibm-runtime >= 0.49 is required"); ok = False
    if info["qiskit_aer"] is None:
        warnings.warn("qiskit-aer is required"); ok = False
    print(f"  reference data: {'ok' if os.path.exists(REF_PATH) else 'MISSING grader_refs.npz'}, "
          f"{'ok' if os.path.exists(MPS_PATH) else 'MISSING mps_reference_L34.npz'}; submission dir: {SUBMISSION_DIR}")
    print("  environment OK" if ok else "  environment has problems (see warnings)")
    info["ok"] = ok
    return info


def summary() -> dict:
    """Print the score table from submission/score.json."""
    scores = _load_scores()
    total = 0.0; total_max = 0; bonus = 0.0
    print(f"{'exercise':10s} {'points':>8s}   detail")
    for ex in MAX_POINTS:
        if ex in scores:
            p = scores[ex]["points"]; m = scores[ex]["max"]
            print(f"{ex:10s} {p:5.1f}/{m:<3d}  {scores[ex]['detail'][:90]}")
            if ex.startswith("B"):
                bonus += p
            else:
                total += p; total_max += m
        elif not ex.startswith("B"):
            print(f"{ex:10s} {'--':>8s}   (not graded)")
    print(f"{'TOTAL':10s} {total:5.1f}/100 graded so far out of {total_max} autograded points; bonus {bonus:.1f}/6 "
          f"(ex 5 = 10 points by the judges)")
    return {"total": total, "graded_max": total_max, "bonus": bonus}


# ---------------------------------------------------------------------------
# Part 0 -- warm-up (function-level, other L)
# ---------------------------------------------------------------------------
def grade_ex0_1(chiral_condensate_observables) -> float:
    """chi_j = (-1)^j Z_j + I at L in {5, 7} (odd L: cannot be copied from the L=34 arrays)."""
    _require_callable("chiral_condensate_observables", chiral_condensate_observables)
    from qiskit.quantum_info import SparsePauliOp
    pts = 0.0; msgs = []
    for L in (5, 7):
        obs, err = _call(chiral_condensate_observables, L)
        if err:
            msgs.append(f"L={L}: raised {err}"); continue
        try:
            obs = list(obs)
        except TypeError:
            msgs.append(f"L={L}: must return a list of SparsePauliOp"); continue
        if len(obs) != 2 * L:
            msgs.append(f"L={L}: expected {2 * L} observables, got {len(obs)}"); continue
        ref = _chi_observables(L)
        bad = None
        for j, (o, ro) in enumerate(zip(obs, ref)):
            if not isinstance(o, SparsePauliOp):
                bad = f"element {j} is {type(o).__name__}, expected SparsePauliOp"; break
            d = _dict_maxdiff(_pauli_dict(o, False), _pauli_dict(ro, False))
            if d > 1e-9:
                bad = f"chi_{j} differs from (-1)^j Z_j + I (max coeff diff {d:.2e}; remember Qiskit little-endian strings)"; break
        if bad:
            msgs.append(f"L={L}: {bad}")
        else:
            pts += 1.0
    detail = "chi_j = (-1)^j Z_j + I verified at L=5,7" if pts == 2 else "; ".join(msgs)
    return _record("ex0.1", pts, detail)


def grade_ex0_2(RXYplus, RXYminus) -> float:
    """RXYplus(th) = exp(-i th/2 (XY+YX)), RXYminus(th) = exp(+i th/2 (XY-YX)) at a seeded random angle."""
    _require_callable("RXYplus", RXYplus); _require_callable("RXYminus", RXYminus)
    from qiskit.quantum_info import Operator
    from scipy.linalg import expm
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]])
    two = lambda A, B: np.kron(B, A)  # noqa: E731  (little-endian: qubit 0 rightmost)
    rng = np.random.default_rng(20261)
    th = float(rng.uniform(0.3, 1.2))
    pts = 0.0; msgs = []
    for name, fn, gen, sign in (("RXYplus", RXYplus, two(X, Y) + two(Y, X), -1), ("RXYminus", RXYminus, two(X, Y) - two(Y, X), +1)):
        g, err = _call(fn, th)
        if err:
            msgs.append(f"{name}: raised {err}"); continue
        try:
            U = Operator(g).data
        except Exception as e:  # noqa: BLE001
            msgs.append(f"{name}: not convertible to a 2-qubit Operator ({e})"); continue
        if U.shape != (4, 4):
            msgs.append(f"{name}: must act on 2 qubits"); continue
        ok, e = _operator_close(U, expm(sign * 1j * th / 2 * gen), 1e-8)
        if ok:
            pts += 1.0
        else:
            hint = ""
            for s2, lab in ((-sign, "opposite sign"), (2 * sign, "double angle"), (-2 * sign, "double angle, opposite sign")):
                if _operator_close(U, expm(s2 * 1j * th / 2 * gen), 1e-8)[0]:
                    hint = f" (looks like {lab})"
            msgs.append(f"{name}(theta) != exp({'-' if sign < 0 else '+'}i theta/2 ({'XY+YX' if sign < 0 else 'XY-YX'})) (max dev {e:.2e}){hint}")
    detail = f"both 2-qubit rotations match their generators at theta={th:.4f}" if pts == 2 else "; ".join(msgs)
    return _record("ex0.2", pts, detail, extra={"theta": th})


def _prep_state(fn, name: str, L: int, *args):
    """Call a state-preparation function and return (statevector, error message)."""
    QC = _qc_types()
    qc, err = _call(fn, L, *args)
    if err:
        return None, f"L={L}: raised {err}"
    if not isinstance(qc, QC):
        return None, f"L={L}: must return a QuantumCircuit"
    if qc.num_qubits != 2 * L:
        return None, f"L={L}: circuit has {qc.num_qubits} qubits, expected {2 * L} (does {name} ignore L?)"
    try:
        return _statevector(_strip_measurements(qc)), None
    except Exception as e:  # noqa: BLE001
        return None, f"L={L}: the circuit cannot be simulated as a statevector ({type(e).__name__}: {str(e)[:80]}; no measurements/resets inside)"


def grade_ex0_3(prep_vacuum) -> float:
    """prep_vacuum(L, 0.30738, -0.04059) at L=6, 8 vs the reference statevector fingerprints (random projections)."""
    _require_callable("prep_vacuum", prep_vacuum)
    pts = 0.0; msgs = []; dists = {}
    for L in (6, 8):
        psi, err = _prep_state(prep_vacuum, "prep_vacuum", L, VACUUM_THETA_OV_1, VACUUM_THETA_OV_3)
        if err:
            msgs.append(err); continue
        ok, d = _fp_match(psi, f"fp_vacuum_L{L}"); dists[L] = d
        if ok:
            pts += 1.0
        else:
            msgs.append(f"L={L}: fingerprint distance from the reference vacuum = {d:.2e} (0 for the right state)")
    if pts == 1 and dists.get(6, 1) < 1e-6:
        msgs.append("L=6 is right but L=8 is not: check the interior R^XY_-(theta) layer of OV_3 (k in range(1, L-1)) and the strong-coupling vacuum X gates on even sites")
    detail = f"vacuum statevectors match at L=6,8 (fingerprint distances {dists.get(6, 0):.1e}, {dists.get(8, 0):.1e})" if pts == 2 else "; ".join(msgs)
    return _record("ex0.3", pts, detail, extra={"fingerprint_distances": dists})


def grade_ex0_4(prep_wave) -> float:
    """prep_wave(L, th1, th3, th11, th22) at L=6, 8 vs the reference statevector fingerprints."""
    _require_callable("prep_wave", prep_wave)
    pts = 0.0; msgs = []; dists = {}
    for L in (6, 8):
        psi, err = _prep_state(prep_wave, "prep_wave", L, VACUUM_THETA_OV_1, VACUUM_THETA_OV_3, WAVE_THETA_O_11, WAVE_THETA_O_22)
        if err:
            msgs.append(err); continue
        ok, d = _fp_match(psi, f"fp_wave_L{L}"); dists[L] = d
        okv, _ = _fp_match(psi, f"fp_vacuum_L{L}")
        if ok:
            pts += 1.0
        elif okv:
            msgs.append(f"L={L}: this is the vacuum -- the O_11 / O_22 wavepacket layers are missing")
        else:
            msgs.append(f"L={L}: fingerprint distance from the reference wavepacket = {d:.2e} (O_22 acts on qubits L-2..L+1: R+(-pi/2)[L-1,L], R+(-th)[L-2,L-1], R+(-th)[L,L+1], R+(pi/2)[L-1,L])")
    detail = f"wavepacket statevectors match at L=6,8 (fingerprint distances {dists.get(6, 0):.1e}, {dists.get(8, 0):.1e})" if pts == 2 else "; ".join(msgs)
    return _record("ex0.4", pts, detail, extra={"fingerprint_distances": dists})


def grade_ex0_5(trotter_step, evolve_circuits, prep_wave) -> float:
    """(1) second-order ratio test of one Trotter step vs expm(-i dt H_trunc) at L=6 (phase-aligned
    error on 3 random states, dt=0.4/0.2/0.1, ratios in [6,10]); (0.5) t=2 physics circuit == pinned
    Fig.-8 ordering; (0.5) t=4 mitigation circuit returns to the initial state."""
    _require_callable("trotter_step", trotter_step); _require_callable("evolve_circuits", evolve_circuits)
    _require_callable("prep_wave", prep_wave)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    r = _refs(); L = 6; n = 2 * L
    psis = ex05_random_states(n)
    evolved = r["ex05_evolved"]                     # (3 dt, 3 states, 2^n): expm(-i dt H_trunc) psi, stored by the organizer
    pts = 0.0; msgs = []; errs = []
    for i_dt, dt in enumerate(EX05_DTS):
        qc, err = _call(trotter_step, QuantumCircuit(n), L, dt, M_DEFAULT, G_DEFAULT)
        if err or not isinstance(qc, QuantumCircuit):
            msgs.append(f"trotter_step raised/returned wrong type: {err}"); errs = None; break
        if qc.num_qubits != n:
            msgs.append(f"trotter_step(QuantumCircuit({n}), L={L}, ...) returned a {qc.num_qubits}-qubit circuit (does trotter_step ignore L?)"); errs = None; break
        e = 0.0
        try:
            for i_p, p in enumerate(psis):
                out = Statevector(p).evolve(_strip_measurements(qc)).data
                e = max(e, _phase_aligned_maxdiff(out, evolved[i_dt, i_p]))
        except Exception as ex:  # noqa: BLE001
            msgs.append(f"trotter_step circuit could not be simulated ({type(ex).__name__}: {str(ex)[:80]})"); errs = None; break
        errs.append(e)
    ratios = None
    if errs:
        ratios = (errs[0] / max(errs[1], 1e-300), errs[1] / max(errs[2], 1e-300))
        if all(6.0 <= x <= 10.0 for x in ratios) and errs[2] < 1e-3:
            pts += 1.0
        elif all(3.0 <= x <= 5.0 for x in ratios):
            msgs.append(f"ratio test: error ratios {ratios[0]:.2f}, {ratios[1]:.2f} ~ 4 -> first-order step (symmetrize: kinetic half-steps before AND after H_el, H_m)")
        elif errs[0] > 0.3:
            msgs.append(f"ratio test: step error {errs[0]:.2e} at dt=0.4 -- the step does not approximate exp(-i dt H) at all (check H_m Rz signs (-1)^j m dt, RXXplus angle dt/4, electric Rz layer)")
        else:
            msgs.append(f"ratio test: error ratios {ratios[0]:.2f}, {ratios[1]:.2f} not in [6,10] (errors {errs[0]:.2e}, {errs[1]:.2e}, {errs[2]:.2e})")
    # physics circuit (pinned ordering, fingerprint of the t=2 state) + mitigation circuit return + gate accounting
    qi, err = _call(prep_wave, L, VACUUM_THETA_OV_1, VACUUM_THETA_OV_3, WAVE_THETA_O_11, WAVE_THETA_O_22)
    init_ok = False
    if err is None and isinstance(qi, QuantumCircuit) and qi.num_qubits == n:
        try:
            init_ok = _fp_match(_statevector(_strip_measurements(qi)), "fp_wave_L6")[0]
        except Exception:  # noqa: BLE001
            init_ok = False
    if not init_ok:
        msgs.append("prep_wave(6) does not match the reference (fix ex0.4 first): the Fig.-8 ordering check is skipped; the return check uses the |0..0> state")
        qi = QuantumCircuit(n)
    d_phys = float("inf"); fid_mit = 0.0; n2_phys = n2_mit = None
    res, err = _call(evolve_circuits, qi.copy(), L, 2.0, M_DEFAULT, G_DEFAULT)
    if err or not (isinstance(res, (tuple, list)) and len(res) == 2 and all(isinstance(c, QuantumCircuit) for c in res)):
        msgs.append(f"evolve_circuits must return (qc, qc_mitig): {err or 'wrong return type'}")
    else:
        if init_ok:
            try:
                ok_p, d_phys = _fp_match(_statevector(_strip_measurements(res[0])), "fp_physics_t2_L6")
            except Exception as ex:  # noqa: BLE001
                ok_p = False; msgs.append(f"physics circuit could not be simulated ({type(ex).__name__})")
            if ok_p:
                pts += 0.5
            elif errs and ratios and all(6.0 <= x <= 10.0 for x in ratios):
                msgs.append(f"valid second-order step but not the Fig. 8 ordering (odd bonds (1,2),(3,4),.. first in the first kinetic half-step, even bonds first in the second) -- the hardware reference assumes Fig. 8 (fingerprint distance {d_phys:.2e})")
            else:
                msgs.append(f"t=2 physics circuit differs from the reference (fingerprint distance {d_phys:.2e}; n_steps = 2 ceil(t/2), dt = t/n_steps)")
        res4, err4 = _call(evolve_circuits, qi.copy(), L, 4.0, M_DEFAULT, G_DEFAULT)
        if err4 or not (isinstance(res4, (tuple, list)) and len(res4) == 2 and all(isinstance(c, QuantumCircuit) for c in res4)):
            msgs.append(f"evolve_circuits(t=4) failed: {err4 or 'wrong return type'}")
        else:
            qp, qm = res4
            try:
                fid_mit = _fidelity(_statevector(_strip_measurements(qm)), _statevector(_strip_measurements(qi)))
                n2_phys, n2_mit = _n2q(qp), _n2q(qm)
            except Exception as ex:  # noqa: BLE001
                msgs.append(f"mitigation circuit could not be simulated ({type(ex).__name__})")
            count_ok = n2_phys is not None and n2_phys > 0 and abs(n2_mit - n2_phys) <= 0.01 * n2_phys
            if fid_mit > 1 - 1e-9 and count_ok:
                pts += 0.5
            elif fid_mit > 1 - 1e-9:
                msgs.append(f"mitigation circuit (t=4) returns to the initial state but has {n2_mit} 2q gates vs {n2_phys} in the physics circuit: it must have the SAME gate content (n_steps/2 forward with +dt, then n_steps/2 with -dt)")
            else:
                msgs.append(f"mitigation circuit (t=4) does not return to the initial state (fidelity {fid_mit:.6f}): n_steps/2 forward with +dt, then n_steps/2 with -dt")
    detail = (f"2nd-order step (ratios {ratios[0]:.2f}, {ratios[1]:.2f}), Fig.-8 ordering (fingerprint distance {d_phys:.1e}), mitigation returns (F={fid_mit:.10f}, {n2_mit} 2q gates as the physics circuit)"
              if pts == 2 else "; ".join(msgs))
    return _record("ex0.5", pts, detail, extra={"errors": errs, "ratios": ratios, "fp_distance_physics": d_phys, "fid_mitig": fid_mit, "n2q_physics_t4": n2_phys, "n2q_mitig_t4": n2_mit})


# ---------------------------------------------------------------------------
# Part 1 -- physics you can verify
# ---------------------------------------------------------------------------
def grade_ex1_1(schwinger_hamiltonian, electric_hamiltonian_truncated, E0_L8) -> float:
    """(2) full H Pauli coefficients (identity excluded) at a seeded L and at L=8; (1) H_el^(1) at the
    same L; (1) [H, Q_tot] = 0 and <Q_tot^2> = 0 on the reference wavepacket; (1) E0(L=8)."""
    _require_callable("schwinger_hamiltonian", schwinger_hamiltonian)
    _require_callable("electric_hamiltonian_truncated", electric_hamiltonian_truncated)
    if not isinstance(E0_L8, (int, float, np.floating, np.integer)):
        raise TypeError("E0_L8 must be a float")
    from qiskit.quantum_info import SparsePauliOp, Statevector
    r = _refs()
    rng = np.random.default_rng(1101)
    Ls = [int(rng.choice([4, 6])), 8]
    pts = 0.0; msgs = []
    # full Hamiltonian
    ok_full = 0
    for L in Ls:
        H, err = _call(schwinger_hamiltonian, L, M_DEFAULT, G_DEFAULT)
        if err or not isinstance(H, SparsePauliOp):
            msgs.append(f"schwinger_hamiltonian(L={L}) -> {err or type(H).__name__}, expected SparsePauliOp"); continue
        if H.num_qubits != 2 * L:
            msgs.append(f"schwinger_hamiltonian(L={L}) acts on {H.num_qubits} qubits, expected {2 * L}"); continue
        d = _dict_maxdiff(_pauli_dict(H), _ref_pauli_dict(f"H_full_labels_L{L}", f"H_full_coeffs_L{L}"))
        if d < 1e-9:
            ok_full += 1
        else:
            dt = _dict_maxdiff(_pauli_dict(H), _ref_pauli_dict(f"H_trunc_labels_L{L}", f"H_trunc_coeffs_L{L}"))
            hint = " (this is the TRUNCATED H_el^(1); the full model needs (sum_{k<=j} Q_k)^2 with open boundaries)" if dt < 1e-9 else ""
            # diagnose electric constant / mass sign
            ref = _ref_pauli_dict(f"H_full_labels_L{L}", f"H_full_coeffs_L{L}")
            mine = _pauli_dict(H)
            zz = [k for k in ref if k.count("Z") == 2]
            z1 = [k for k in ref if k.count("Z") == 1]
            hint2 = ""
            if all(abs(mine.get(k, 0) - ref[k]) < 1e-9 for k in zz) and any(abs(mine.get(k, 0) - ref[k]) > 1e-9 for k in z1):
                hint2 = " (ZZ terms right, single-Z terms wrong: check the (-1)^k I part of Q_k, which produces the electric single-Z terms, and the mass sign (-1)^j)"
            msgs.append(f"schwinger_hamiltonian(L={L}): max coefficient difference {d:.2e} (identity term excluded){hint}{hint2}")
    pts += {0: 0.0, 1: 1.0, 2: 2.0}[ok_full]
    # truncated electric
    L = Ls[0]
    Hel, err = _call(electric_hamiltonian_truncated, L, G_DEFAULT)
    if err or not isinstance(Hel, SparsePauliOp):
        msgs.append(f"electric_hamiltonian_truncated(L={L}) -> {err or type(Hel).__name__}")
    else:
        Hm = _ref_pauli_dict(f"H_full_labels_L{L}", f"H_full_coeffs_L{L}")  # noqa: F841
        ref_el = {k: v for k, v in _ref_pauli_dict(f"H_trunc_labels_L{L}", f"H_trunc_coeffs_L{L}").items() if set(k) <= {"I", "Z"}}
        # remove the mass single-Z part: mass contributes (-1)^j m/2 on single Z
        n = 2 * L
        for j in range(n):
            lab = "I" * (n - 1 - j) + "Z" + "I" * j
            ref_el[lab] = ref_el.get(lab, 0) - (-1) ** j * M_DEFAULT / 2
        ref_el = {k: v for k, v in ref_el.items() if abs(v) > 1e-12}
        d = _dict_maxdiff(_pauli_dict(Hel), ref_el)
        if d < 1e-9:
            pts += 1.0
        else:
            msgs.append(f"electric_hamiltonian_truncated(L={L}): max coefficient difference {d:.2e} vs H_el^(Q=0)(1) of the notebook equation")
    # charge sector
    H8, err = _call(schwinger_hamiltonian, 6, M_DEFAULT, G_DEFAULT)
    if err is None and isinstance(H8, SparsePauliOp) and H8.num_qubits == 12:
        n = 12
        Q = 0 * SparsePauliOp("I" * n)
        for k in range(n):
            lab = "I" * (n - 1 - k) + "Z" + "I" * k
            Q += SparsePauliOp.from_list([(lab, -0.5), ("I" * n, -0.5 * (-1) ** k)])
        Q = Q.simplify()
        comm = (H8 @ Q - Q @ H8).simplify()
        cnorm = float(np.max(np.abs(comm.coeffs))) if len(comm.coeffs) else 0.0
        sv = Statevector(r["sv_wave_L6"])
        q2 = float(np.real(sv.expectation_value((Q @ Q).simplify())))
        if cnorm < 1e-9 and q2 < 1e-8:
            pts += 1.0
        else:
            msgs.append(f"charge sector: ||[H,Q_tot]|| = {cnorm:.2e}, <Q_tot^2>_wave = {q2:.2e}")
    else:
        msgs.append("charge-sector check skipped (schwinger_hamiltonian(6) unusable)")
    # E0
    e_ref = float(r["E0_L8"])
    if abs(float(E0_L8) - e_ref) < 1e-4:
        pts += 1.0
    elif abs(float(E0_L8) - (e_ref - 2 * 8 * M_DEFAULT / 2)) < 1e-4:
        pts += 0.5
        msgs.append(f"E0_L8 = {float(E0_L8):.5f} drops the +m/2*I per site (16 sites x 0.25 = 4.0); the stated convention gives {e_ref:.5f}")
    else:
        msgs.append(f"E0_L8 = {float(E0_L8):.5f}, expected {e_ref:.5f} (full H, +m/2*I per site included, eigsh which='SA')")
    detail = (f"H (identity excluded) and H_el^(1) match at L={Ls[0]},8; [H,Q]=0, <Q^2>=0; E0(L=8) = {e_ref:.5f} (convention: +m/2*I per site)"
              if pts == 5 else "; ".join(msgs))
    return _record("ex1.1", pts, detail, extra={"E0_L8": float(E0_L8), "Ls": Ls})


def grade_ex1_2(electric_layer, truncation_shift) -> float:
    """(2) electric_layer(L, t, g) == exp(-i t H_el^(1)) on 3 random states at L=4, 6 (infidelity <
    1e-10); (2) truncation_shift = max_j |X_full - X_trunc| at L=8, t=4 within 0.002 of the reference
    (a 16-element array X_full - X_trunc is also accepted)."""
    _require_callable("electric_layer", electric_layer)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from scipy.sparse.linalg import expm_multiply
    r = _refs(); rng = np.random.default_rng(1202)
    pts = 0.0; msgs = []; infids = {}
    for L in (4, 6):
        n = 2 * L; t = float(rng.uniform(0.15, 0.6))
        qc, err = _call(electric_layer, L, t, G_DEFAULT)
        if err or not isinstance(qc, QuantumCircuit) or qc.num_qubits != n:
            msgs.append(f"electric_layer(L={L}) -> {err or 'wrong type/size'}"); continue
        Hel = SparsePauliOp([str(l) for l in r[f"Hel_trunc_labels_L{L}"]], r[f"Hel_trunc_coeffs_L{L}"]).to_matrix(sparse=True)
        worst = 0.0
        for _ in range(3):
            v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n); v /= np.linalg.norm(v)
            out = Statevector(v).evolve(qc).data
            ex = expm_multiply(-1j * t * Hel, v)
            worst = max(worst, 1 - _fidelity(out, ex))
        infids[L] = worst
        if worst < 1e-10:
            pts += 1.0
        else:
            msgs.append(f"electric_layer(L={L}, t={t:.3f}): infidelity {worst:.2e} vs exp(-i t H_el^(1)) (Rz layer: Rz(g^2 t) on 2k, Rz(g^2 t/2) on 2k+1 for k<L/2-1, Rz(g^2 t/2) on L-2, Rz(-g^2 t/2) on L+1, Rz(-g^2 t/2) on L+2k, Rz(-g^2 t) on L+2k+1 for 1<=k<L/2; then trotter_step_electric_2q)")
    ref = float(r["truncation_shift_L8_t4"]); ref_arr = r["truncation_shift_array_L8_t4"]
    ts = np.asarray(truncation_shift, dtype=float)
    if ts.ndim == 0:
        if abs(float(ts) - ref) < 0.002:
            pts += 2.0
        else:
            msgs.append(f"truncation_shift = {float(ts):.4f}, expected max_j |X_full - X_trunc| = {ref:.4f} at L=8, t=4 (exact expm of both Hamiltonians on prep_wave AND prep_vacuum, X = chi_wave - chi_vac)")
    elif ts.shape == (16,):
        d = min(np.max(np.abs(ts - ref_arr)), np.max(np.abs(ts + ref_arr)))
        if d < 0.002:
            pts += 2.0
        else:
            msgs.append(f"truncation_shift array deviates by {d:.4f} from X_full - X_trunc (L=8, t=4)")
    else:
        raise TypeError("truncation_shift must be a float (max_j |X_full - X_trunc|) or a 16-element array")
    detail = f"electric layer exact at L=4,6 (infidelities {infids.get(4, 0):.1e}, {infids.get(6, 0):.1e}); truncation shift {ref:.4f} confirmed" if pts == 4 else "; ".join(msgs)
    return _record("ex1.2", pts, detail, arrays={"truncation_shift": ts}, extra={"infidelities": infids})


def grade_ex1_3(vqe_fidelity, vqe_energy_gap) -> float:
    """vqe_fidelity {L: |<gs|ADAPT>|^2} and vqe_energy_gap {L: E_ADAPT - E0} at L=4,6,8 within 1e-3."""
    _require_type("vqe_fidelity", vqe_fidelity, dict); _require_type("vqe_energy_gap", vqe_energy_gap, dict)
    r = _refs(); pts = 0.0; msgs = []
    for L in (4, 6, 8):
        f = vqe_fidelity.get(L, vqe_fidelity.get(str(L)))
        g = vqe_energy_gap.get(L, vqe_energy_gap.get(str(L)))
        fr = float(r[f"vqe_fid_L{L}"]); gr = float(r[f"E_adapt_L{L}"] - r[f"E0_L{L}"])
        if f is None or g is None:
            msgs.append(f"L={L} missing"); continue
        okf = abs(float(f) - fr) < 1e-3
        okg = abs(abs(float(g)) - gr) < 1e-3
        pts += 0.5 * okf + 0.5 * okg
        if not okf:
            msgs.append(f"vqe_fidelity[{L}] = {float(f):.5f}, expected {fr:.5f} (overlap with eigsh ground state of the FULL H)")
        if not okg:
            msgs.append(f"vqe_energy_gap[{L}] = {float(g):.5f}, expected {gr:.5f} (E_ADAPT - E0, both with +m/2*I per site)")
    detail = "fidelities 0.9961/0.9945/0.9929 and energy gaps confirmed at L=4,6,8" if pts == 3 else "; ".join(msgs)
    return _record("ex1.3", pts, detail, arrays={"vqe_fidelity": json.dumps(_jsonable(vqe_fidelity)), "vqe_energy_gap": json.dumps(_jsonable(vqe_energy_gap))})


def _norm_table(trotter_table: dict) -> dict[tuple[float, float], float]:
    out = {}
    for k, v in trotter_table.items():
        if isinstance(k, (tuple, list)) and len(k) == 2:
            out[(round(float(k[0]), 6), round(float(k[1]), 6))] = float(v)
        elif isinstance(k, str):
            parts = k.replace("(", "").replace(")", "").replace("t=", "").replace("dt=", "").replace(",", " ").split()
            if len(parts) == 2:
                out[(round(float(parts[0]), 6), round(float(parts[1]), 6))] = float(v)
    return out


def grade_ex1_4(trotter_table, richardson_error, dominant_error) -> float:
    """(2) six entries {(t, dt): max_j |X_trotter - X_exact|} within 0.003; (1) ratios e(1)/e(0.5), e(0.5)/e(0.25)
    in [3, 5.5] at both t; (1) richardson_error in (0, 0.006); (1) dominant_error == 'trotter'."""
    _require_type("trotter_table", trotter_table, dict)
    if not isinstance(richardson_error, (int, float, np.floating)):
        raise TypeError("richardson_error must be a float")
    _require_type("dominant_error", dominant_error, str)
    r = _refs(); tab = _norm_table(trotter_table)
    pts = 0.0; msgs = []; n_ok = 0
    for t in (2.0, 4.0):
        for dt in (1.0, 0.5, 0.25):
            ref = float(r[f"trotter_table_t{int(t)}_dt{dt:g}"])
            v = tab.get((t, dt))
            if v is None:
                msgs.append(f"entry ({t:g}, {dt:g}) missing"); continue
            if abs(v - ref) < 0.003:
                n_ok += 1
            else:
                msgs.append(f"({t:g},{dt:g}): {v:.4f} vs reference {ref:.4f}")
    pts += 2.0 * n_ok / 6
    ratios = {}
    ok_r = True
    for t in (2.0, 4.0):
        try:
            r1 = tab[(t, 1.0)] / tab[(t, 0.5)]; r2 = tab[(t, 0.5)] / tab[(t, 0.25)]
        except (KeyError, ZeroDivisionError):
            ok_r = False; continue
        ratios[t] = (r1, r2)
        if not (3.0 <= r1 <= 5.5 and 3.0 <= r2 <= 5.5):
            ok_r = False; msgs.append(f"t={t:g}: error ratios {r1:.2f}, {r2:.2f} not in [3, 5.5] (global 2nd-order error scales as dt^2)")
    if ok_r and ratios:
        pts += 1.0
    if 0 < float(richardson_error) < 0.006:
        pts += 1.0
    else:
        msgs.append(f"richardson_error = {float(richardson_error):.4f}: X_R = (4 X(dt=0.25) - X(dt=0.5))/3 should be within 0.006 of the exact profile (reference {float(r['richardson_error_t4']):.1e} at t=4)")
    if dominant_error.strip().lower().startswith("trotter"):
        pts += 1.0
    else:
        msgs.append(f"dominant_error = '{dominant_error}': at dt=1 the Trotter error ({float(r['trotter_table_t4_dt1']):.3f}) exceeds the truncation shift ({float(r['truncation_shift_L8_t4']):.3f})")
    detail = "Trotter table, dt^2 scaling, Richardson and dominant error all confirmed" if pts == 5 else "; ".join(msgs)
    return _record("ex1.4", pts, detail, arrays={"trotter_table": json.dumps({f"{k[0]:g},{k[1]:g}": v for k, v in tab.items()}), "richardson_error": float(richardson_error)}, extra={"ratios": ratios, "dominant_error": dominant_error})


def grade_ex1_5(mps_err, mps_seconds, X_bond40) -> float:
    """(1) mps_err {bond: max|X(bond) - X(64)|} decreasing (one 10 % wobble allowed); (1) bond-8 err > 0.02
    and bond-20 err < 0.02 (why the QDC bond-20 file is unusable); (1) X_bond40 within 0.01 of the organizer bond-64 X."""
    _require_type("mps_err", mps_err, dict); _require_type("mps_seconds", mps_seconds, dict)
    X40 = np.asarray(X_bond40, dtype=float)
    if X40.shape != (68,):
        raise TypeError("X_bond40 must be a 68-element array (chi_wave - chi_vacuum at t=8, bond 40)")
    errs = {int(k): float(v) for k, v in mps_err.items()}
    bonds = sorted(errs)
    pts = 0.0; msgs = []
    if len(bonds) < 3:
        msgs.append("mps_err needs at least 3 bond dimensions (e.g. 8, 20, 40)")
    else:
        wobbles = 0; bad = False
        for a, b in zip(bonds[:-1], bonds[1:]):
            if errs[b] > errs[a]:
                if errs[b] <= 1.1 * errs[a] and wobbles == 0:
                    wobbles += 1
                else:
                    bad = True
        if not bad:
            pts += 1.0
        else:
            msgs.append(f"mps_err is not monotonically decreasing with the bond dimension: {errs}")
    if 8 in errs and 20 in errs:
        if errs[8] > 0.02 and errs[20] < 0.02:
            pts += 1.0
        else:
            msgs.append(f"expected err(bond 8) > 0.02 and err(bond 20) < 0.02, got {errs[8]:.4f}, {errs[20]:.4f} (compare with the bond-64 organizer profile)")
    else:
        msgs.append("mps_err must contain bonds 8 and 20")
    try:
        Xref = x_ref_t8(64)
        d = float(np.max(np.abs(X40 - Xref)))
        if d < 0.01:
            pts += 1.0
        else:
            hint = " (looks like the odd-first/even-first kinetic ordering was swapped: see ex0.5)" if 0.015 < d < 0.03 else ""
            msgs.append(f"max |X_bond40 - X_bond64| = {d:.4f} > 0.01{hint}")
    except FileNotFoundError as e:
        msgs.append(str(e)); d = None
    secs = {str(k): float(v) for k, v in mps_seconds.items()}
    detail = f"bond convergence ok ({errs}); X(bond 40) within {d:.4f} of the bond-64 reference; wall times {secs}" if pts == 3 else "; ".join(msgs)
    return _record("ex1.5", pts, detail, arrays={"X_bond40": X40, "mps_err": json.dumps(errs), "mps_seconds": json.dumps(secs)})


# ---------------------------------------------------------------------------
# Part 2 -- engineering the 68-qubit experiment
# ---------------------------------------------------------------------------
def _as_observable_list(oset) -> list:
    """list/tuple/ndarray of SparsePauliOp, or an ObservablesArray (dict entries) -> list of SparsePauliOp."""
    from qiskit.quantum_info import SparsePauliOp
    if isinstance(oset, SparsePauliOp):
        return [oset]
    if isinstance(oset, np.ndarray):
        items = list(oset.ravel())
    elif isinstance(oset, (list, tuple)):
        items = list(oset)
    elif hasattr(oset, "tolist"):
        items = list(np.asarray(oset.tolist(), dtype=object).ravel())
    else:
        items = list(oset)
    out = []
    for o in items:
        if isinstance(o, dict):
            o = SparsePauliOp.from_list(list(o.items()))
        out.append(o)
    return out


def grade_ex2_1(circuits_all_isa, observables_isa, backend) -> float:
    """(1) all 4 circuits ISA for backend.target; (1) identical initial/final layouts forming a simple path;
    (1) 2q-depth of the physics circuit in [200, 300] and observables_isa == chi_j.apply_layout(layout)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    if not isinstance(circuits_all_isa, (list, tuple)) or len(circuits_all_isa) != 4 or not all(isinstance(c, QuantumCircuit) for c in circuits_all_isa):
        raise TypeError("circuits_all_isa must be a list of 4 QuantumCircuits [wave, wave_mitig, vacuum, vacuum_mitig]")
    if not hasattr(backend, "target"):
        raise TypeError("backend must be a BackendV2 (e.g. FakeKingston())")
    pts = 0.0; msgs = []
    isa_ok = True
    for i, c in enumerate(circuits_all_isa):
        ok, why = _is_isa(c, backend)
        if not ok:
            isa_ok = False; msgs.append(f"circuit {i}: {why}"); break
    if isa_ok:
        pts += 1.0
    lays = [_layouts(c) for c in circuits_all_isa]
    lay0 = lays[0][0]
    if any(l[0] is None for l in lays):
        msgs.append("circuits have no layout (transpile with generate_preset_pass_manager(backend=...))")
    elif any(l[0] != lay0 or l[1] != lay0 for l in lays):
        msgs.append("initial/final layouts differ between the 4 circuits (transpile all with initial_layout=<layout of the physics circuit>; a swap-free chain keeps final == initial)")
    else:
        summ = target_summary(backend)
        ok, why = chain_validity(summ, lay0, 68) if len(lay0) == 68 else (False, f"layout has {len(lay0)} qubits")
        if ok or (len(lay0) == 68 and "dead" in why):
            pts += 1.0
            if not ok:
                msgs.append(f"layout warning: {why}")
        else:
            msgs.append(f"layout is not a simple path on the coupling map: {why}")
    d2 = _two_qubit_depth(circuits_all_isa[0])
    depth_ok = 200 <= d2 <= 300
    if not depth_ok:
        msgs.append(f"2q-depth of the physics circuit is {d2}, expected 200-300 (swap-free chain, optimization_level 3)")
    obs_ok = False
    try:
        obs_sets = observables_isa
        if isinstance(observables_isa, (list, tuple)) and len(observables_isa) == 4 and all(isinstance(o, (list, tuple, np.ndarray)) or hasattr(o, "tolist") for o in observables_isa) \
                and not isinstance(observables_isa[0], SparsePauliOp):
            obs_sets = list(observables_isa)
        else:
            obs_sets = [observables_isa] * 4
        logical = _chi_observables(34)
        obs_ok = True
        for c, oset in zip(circuits_all_isa, obs_sets):
            oset = _as_observable_list(oset)
            if len(oset) != 68:
                obs_ok = False; msgs.append(f"observables_isa must hold 68 chi_j observables (got {len(oset)})"); break
            if c.layout is None:
                obs_ok = False; break
            for j, (o, lo) in enumerate(zip(oset, logical)):
                if not isinstance(o, SparsePauliOp):
                    obs_ok = False; msgs.append(f"observable {j} is {type(o).__name__}, expected SparsePauliOp"); break
                exp = lo.apply_layout(c.layout)
                if _dict_maxdiff(_pauli_dict(o, False), _pauli_dict(exp, False)) > 1e-9:
                    obs_ok = False; msgs.append(f"observable {j} != chi_{j}.apply_layout(layout) (use SparsePauliOp.apply_layout(qc_isa.layout))"); break
            if not obs_ok:
                break
    except Exception as e:  # noqa: BLE001
        obs_ok = False
        msgs.append(f"observables_isa could not be checked: {e}")
    if depth_ok and obs_ok:
        pts += 1.0
    detail = f"4 ISA circuits on one 68-qubit path {lay0[:3]}..{lay0[-3:]} , 2q-depth {d2}, 68 layout-consistent observables" if pts == 3 else "; ".join(msgs)
    return _record("ex2.1", pts, detail, arrays={"layout": np.asarray(lay0 if lay0 else []), "cz_counts": np.array([c.count_ops().get("cz", 0) for c in circuits_all_isa]), "depth2q": d2},
                   extra={"backend": getattr(backend, "name", "?")})


def grade_ex2_2(n2q_logical, n_cz_physics, evolve_circuits_matched, prep_wave, backend, layout) -> float:
    """(1) n2q_logical within 1 % of the reference; (1) n_cz_physics within 1 % of FakeKingston O3 seed 42
    (half credit for the O1 count); (2) evolve_circuits_matched: transpiled mitigation circuit has >= 0.99 x the
    physics CZ count and per-bond CZ counts within +-2; (1) at L=6 the matched mitigation circuit returns to the initial state."""
    if not isinstance(n2q_logical, (int, np.integer)) or not isinstance(n_cz_physics, (int, np.integer)):
        raise TypeError("n2q_logical and n_cz_physics must be ints")
    _require_callable("evolve_circuits_matched", evolve_circuits_matched); _require_callable("prep_wave", prep_wave)
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from collections import Counter
    r = _refs(); pts = 0.0; msgs = []
    ref2q = int(r["n2q_logical_t8"]); refcz = int(r["n_cz_physics_t8_kingston_O3_seed42"]); refcz1 = int(r["n_cz_physics_t8_kingston_O1_seed42"])
    if abs(int(n2q_logical) - ref2q) <= 0.01 * ref2q:
        pts += 1.0
    else:
        msgs.append(f"n2q_logical = {n2q_logical}, expected {ref2q} +-1 % (count 2-qubit gates of qc_wave.decompose(reps=3), t=8: 8 Trotter steps)")
    if abs(int(n_cz_physics) - refcz) <= 0.01 * refcz:
        pts += 1.0
    elif abs(int(n_cz_physics) - refcz1) <= 0.01 * refcz1:
        pts += 0.5; msgs.append(f"n_cz_physics = {n_cz_physics} is the optimization_level-1 count; level 3 merges the adjacent kinetic layers of consecutive steps -> {refcz}")
    else:
        msgs.append(f"n_cz_physics = {n_cz_physics}, expected {refcz} +-1 % (FakeKingston, optimization_level=3, seed_transpiler=42)")
    # matched circuits at L=34
    lay = [int(q) for q in layout]
    if len(lay) != 68:
        raise TypeError("layout must be a list of 68 physical qubits")
    qi, err = _call(prep_wave, 34, VACUUM_THETA_OV_1, VACUUM_THETA_OV_3, WAVE_THETA_O_11, WAVE_THETA_O_22)
    if err or not isinstance(qi, QuantumCircuit):
        msgs.append(f"prep_wave(34) failed: {err}")
    else:
        res, err = _call(evolve_circuits_matched, qi, 34, 8.0, M_DEFAULT, G_DEFAULT)
        if err or not (isinstance(res, (tuple, list)) and len(res) == 2 and all(isinstance(c, QuantumCircuit) for c in res)):
            msgs.append(f"evolve_circuits_matched must return (qc, qc_mitig): {err or 'wrong type'}")
        else:
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42, initial_layout=lay)
            try:
                isa_p = pm.run(res[0]); isa_m = pm.run(res[1])
                cp = Counter(tuple(sorted(e)) for e in _cz_pairs(isa_p)); cm = Counter(tuple(sorted(e)) for e in _cz_pairs(isa_m))
                np_, nm_ = sum(cp.values()), sum(cm.values())
                diffs = {k: cm[k] - cp[k] for k in set(cp) | set(cm)}
                worst = max(abs(v) for v in diffs.values()) if diffs else 0
                if nm_ >= 0.99 * np_ and worst <= 2:
                    pts += 2.0
                else:
                    msgs.append(f"matched mitigation circuit: {nm_} CZ vs {np_} in the physics circuit, worst per-bond difference {worst} "
                                f"(the transpiler cancels the last forward kinetic layer against the first backward one; protect the turning point, e.g. with a barrier)")
            except Exception as e:  # noqa: BLE001
                msgs.append(f"transpiling the matched circuits failed: {e}")
    # L=6 fidelity
    qi6, err = _call(prep_wave, 6, VACUUM_THETA_OV_1, VACUUM_THETA_OV_3, WAVE_THETA_O_11, WAVE_THETA_O_22)
    if err is None and isinstance(qi6, QuantumCircuit):
        res, err = _call(evolve_circuits_matched, qi6, 6, 4.0, M_DEFAULT, G_DEFAULT)
        if err is None and isinstance(res, (tuple, list)) and len(res) == 2:
            f = _fidelity(_statevector(res[1]), _statevector(qi6))
            if f > 1 - 1e-6:
                pts += 1.0
            else:
                msgs.append(f"matched mitigation circuit at L=6 does not return to the initial state (fidelity {f:.6f})")
    detail = f"gate accounting ({n2q_logical} logical 2q, {n_cz_physics} CZ) and noise-matched mitigation circuit confirmed" if pts == 5 else "; ".join(msgs)
    return _record("ex2.2", pts, detail, arrays={"n2q_logical": int(n2q_logical), "n_cz_physics": int(n_cz_physics), "layout": np.asarray(lay)})


def grade_ex2_3(select_chain, backend, n_qubits: int = 68) -> float:
    """(2) select_chain(backend, n_qubits=68) returns a valid 68-qubit simple path with no dead edge/qubit;
    (3/2/1) cost C = sum_bonds 74 (-ln(1-eps_CZ)) + sum_qubits [150 (-ln(1-eps_sx)) + (-ln(1-eps_RO))] (the notebook's
    expected-error objective) <= 1.05 / 1.15 / 1.30 x the grader's own seeded randomized-DFS baseline (frozen target
    summary).  Runtime capped at 60 s."""
    _require_callable("select_chain", select_chain)
    if not hasattr(backend, "target"):
        raise TypeError("backend must be a BackendV2")
    summ = target_summary(backend)
    chain, err, el, timed_out = _run_with_timeout(select_chain, 60.0, backend, n_qubits=n_qubits)
    if timed_out:
        return _record("ex2.3", 0.0, f"select_chain did not finish within 60 s ({getattr(backend, 'name', '?')})")
    if err:
        return _record("ex2.3", 0.0, f"select_chain raised {err}")
    try:
        chain = [int(q) for q in chain]
    except Exception:
        return _record("ex2.3", 0.0, "select_chain must return a list of 68 ints")
    ok, why = chain_validity(summ, chain, n_qubits)
    if not ok:
        return _record("ex2.3", 0.0, f"invalid chain: {why}", arrays={"chain": np.asarray(chain)})
    pts = 2.0
    c = chain_cost(summ, chain)
    cb, base = _baseline_for(summ, n_qubits)
    ratio = c / cb if cb > 0 else float("inf")
    if ratio <= 1.05:
        pts += 3.0
    elif ratio <= 1.15:
        pts += 2.0
    elif ratio <= 1.30:
        pts += 1.0
    detail = (f"valid path on {getattr(backend, 'name', '?')} ({el:.1f} s); cost {c:.4f} vs baseline {cb:.4f} (ratio {ratio:.3f})")
    if pts < 5:
        detail += " -- a calibration-aware search (randomized DFS / beam search over the expected-error cost 74 CZ/bond + 150 sx/qubit + readout, many restarts) should reach the baseline"
    return _record("ex2.3", pts, detail, arrays={"chain": np.asarray(chain), "cost": c, "baseline_cost": cb, "baseline_chain": np.asarray(base)},
                   extra={"backend": getattr(backend, "name", "?"), "seconds": el, "ratio": ratio})


def _opt_get(options, *path):
    cur = options
    for p in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return cur


def grade_ex2_4(flight_plan, estimator_options, circuits_all_isa) -> float:
    """(1) flight_plan fields + usage model 2 s + 0.45 ms x (4 x twirls x shots) <= 120 s, executions <= 160000,
    chain == layout of the ISA circuits; (1) EstimatorOptions: gate+measure twirling, DD on, resilience_level 0,
    max_execution_time <= 180."""
    _require_type("flight_plan", flight_plan, dict)
    pts = 0.0; msgs = []
    need = ["backend", "chain", "num_randomizations", "shots_per_randomization", "dd_sequence", "predicted_usage_s", "predicted_sigma_X"]
    missing = [k for k in need if k not in flight_plan]
    usage = None
    if missing:
        msgs.append(f"flight_plan missing {missing}")
    else:
        try:
            tw = int(flight_plan["num_randomizations"]); sh = int(flight_plan["shots_per_randomization"])
            execs = 4 * tw * sh
            usage = USAGE_OVERHEAD_S + USAGE_PER_EXECUTION_S * execs
            pu = float(flight_plan["predicted_usage_s"]); sig = float(flight_plan["predicted_sigma_X"])
            chain = [int(q) for q in flight_plan["chain"]]
            ok = True
            if usage > 120:
                ok = False; msgs.append(f"predicted usage {usage:.1f} s > 120 s ({execs} executions)")
            if execs > 160000:
                ok = False; msgs.append(f"{execs} executions > 160000")
            if abs(pu - usage) > 0.25 * usage + 1.0:
                ok = False; msgs.append(f"predicted_usage_s = {pu:.1f} but the usage model gives {usage:.1f} s = 2 + 0.45e-3 x 4 x {tw} x {sh}")
            if not (1e-3 < sig < 0.5):
                ok = False; msgs.append(f"predicted_sigma_X = {sig} is not a plausible per-site uncertainty of X (shot noise / retention)")
            if len(chain) != 68 or len(set(chain)) != 68:
                ok = False; msgs.append("chain must list 68 distinct physical qubits")
            elif circuits_all_isa is not None:
                lay0 = _layouts(circuits_all_isa[0])[0]
                if lay0 is not None and lay0 != chain and lay0 != chain[::-1]:
                    ok = False; msgs.append("flight_plan['chain'] differs from the layout of circuits_all_isa")
            if not isinstance(flight_plan["dd_sequence"], str) or not isinstance(flight_plan["backend"], str):
                ok = False; msgs.append("dd_sequence and backend must be strings")
            if ok:
                pts += 1.0
        except (TypeError, ValueError) as e:
            msgs.append(f"flight_plan fields have wrong types: {e}")
    # options
    o = estimator_options
    opts_ok = True
    if isinstance(o, dict) or o.__class__.__name__ == "EstimatorOptions":
        def _truthy(x):
            return x is True or (isinstance(x, (int, np.integer)) and x == 1)
        checks = {
            "twirling.enable_gates": _truthy(_opt_get(o, "twirling", "enable_gates")),
            "twirling.enable_measure": _truthy(_opt_get(o, "twirling", "enable_measure")),
            "dynamical_decoupling.enable": _truthy(_opt_get(o, "dynamical_decoupling", "enable")),
            "resilience_level == 0": _opt_get(o, "resilience_level") == 0,
        }
        met = _opt_get(o, "max_execution_time")
        checks["max_execution_time <= 180"] = isinstance(met, (int, float)) and 0 < met <= 180
        bad = [k for k, v in checks.items() if not v]
        if bad:
            opts_ok = False; msgs.append(f"estimator_options: {', '.join(bad)} not satisfied")
    else:
        raise TypeError("estimator_options must be an EstimatorOptions or a dict")
    if opts_ok:
        pts += 1.0
    detail = f"flight plan ok (predicted usage {usage:.1f} s) and EstimatorOptions ok" if pts == 2 else "; ".join(msgs)
    return _record("ex2.4", pts, detail, arrays={"flight_plan": json.dumps(_jsonable(flight_plan))}, extra={"usage_model_s": usage})


# ---------------------------------------------------------------------------
# Part 3 -- mitigation you wrote yourself
# ---------------------------------------------------------------------------
def _synthetic_odr(L: int, seed: int):
    """CP-symmetric chi_true in [0,2], f ~ U(0.02,0.95), 10 % dead sites (f=0.001)."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0, L); chi_true = np.concatenate([half, half[::-1]])
    ex_half = rng.uniform(0.0, 0.9, L); chi_exact = np.concatenate([ex_half, ex_half[::-1]])
    f = rng.uniform(0.02, 0.95, 2 * L)
    n_dead = max(3, int(round(0.1 * 2 * L)))
    pair = int(rng.integers(0, L))                      # one fully dead mirror pair -> NaN expected
    single = int(rng.choice([q for q in range(L) if q != pair]))   # one half-dead pair -> live partner only
    dead = [pair, 2 * L - 1 - pair, single]
    others = [q for q in range(2 * L) if q not in dead and (2 * L - 1 - q) not in dead]
    dead += list(rng.choice(others, size=max(0, n_dead - 3), replace=False))
    f[dead] = 0.001
    chi_cal = 1 - f * (1 - chi_exact)
    chi_meas = 1 - f * (1 - chi_true)
    return chi_true, chi_exact, f, chi_cal, chi_meas


def grade_ex3_1(odr_mitigate, odr_uncertainty, odr_bias) -> float:
    """(3) odr_mitigate recovers a CP-symmetric chi_true exactly (1e-9) on synthetic data at L=6,8,34 (NaN
    where both mirror partners are dead); (2) odr_uncertainty within 25 % of a 10^4-sample Monte-Carlo at
    sites with f > 0.2; (2) odr_bias == (1-chi_true)(1 - f_phys/f_cal)."""
    for nm, fn in (("odr_mitigate", odr_mitigate), ("odr_uncertainty", odr_uncertainty), ("odr_bias", odr_bias)):
        _require_callable(nm, fn)
    pts = 0.0; msgs = []
    for i, L in enumerate((6, 8, 34)):
        chi_true, chi_exact, f, chi_cal, chi_meas = _synthetic_odr(L, 3100 + i)
        out, err = _call(odr_mitigate, chi_meas, chi_cal, chi_exact, L, suppression_threshold=0.01)
        if err:
            msgs.append(f"odr_mitigate(L={L}) raised {err}"); continue
        out = np.asarray(out, dtype=float)
        if out.shape != (2 * L,):
            msgs.append(f"odr_mitigate(L={L}) must return 2L values"); continue
        expect = _qdc_odr(chi_meas, chi_cal, chi_exact, L, 0.01)
        nan_ok = np.array_equal(np.isnan(out), np.isnan(expect))
        m = ~np.isnan(expect)
        d = float(np.max(np.abs(out[m] - chi_true[m]))) if m.any() else 0.0
        # noisy replica: shot noise sigma=0.005 on chi and chi_cal; a dead site divided back on its own
        # gives garbage (0.005/0.001 = 5), the mirror-pooled estimate stays within a few sigma/f
        rng_n = np.random.default_rng(3150 + i); sig = 0.005
        chi_n = chi_meas + sig * rng_n.normal(size=2 * L); cal_n = chi_cal + sig * rng_n.normal(size=2 * L)
        out_n, err_n = _call(odr_mitigate, chi_n, cal_n, chi_exact, L, suppression_threshold=0.01)
        noisy_ok = False; dn = float("nan")
        if err_n is None:
            out_n = np.asarray(out_n, dtype=float)
            ref_n = _qdc_odr(chi_n, cal_n, chi_exact, L, 0.01)
            if out_n.shape == (2 * L,) and np.array_equal(np.isnan(out_n), np.isnan(ref_n)):
                mm = ~np.isnan(ref_n)
                f_pool = np.array([np.mean([f[q] for q in (j, 2 * L - 1 - j) if f[q] > 0.01]) if any(f[q] > 0.01 for q in (j, 2 * L - 1 - j)) else np.nan for j in range(2 * L)])
                tol = 8 * sig / f_pool[mm]
                dn = float(np.max(np.abs(out_n[mm] - chi_true[mm]) / tol))
                noisy_ok = dn <= 1.0
        if nan_ok and d < 1e-9 and noisy_ok:
            pts += 1.0
        else:
            hint = ""
            if not nan_ok:
                hint = " (sites where BOTH mirror partners have suppression factor <= threshold must be NaN; a site with one dead partner uses the live partner only)"
            elif d > 1e-9:
                hint = " (pool the two mirror partners: chi_mit = 1 - (1 - mean(chi_sel)) / mean(f_sel), f = (1-chi_cal)/(1-chi_exact))"
            elif not noisy_ok:
                hint = f" (with shot noise the estimate at a dead site must come from its live mirror partner, not from dividing by f=0.001; worst deviation {dn:.1f} x (8 sigma/f))"
            msgs.append(f"odr_mitigate(L={L}): noise-free max deviation from chi_true {d:.2e}, NaN pattern {'ok' if nan_ok else 'wrong'}, noisy replica {'ok' if noisy_ok else 'FAIL'}{hint}")
    # uncertainty
    L = 8
    chi_true, chi_exact, f, chi_cal, chi_meas = _synthetic_odr(L, 3200)
    rng = np.random.default_rng(3201)
    chi_std = rng.uniform(0.01, 0.05, 2 * L); cal_std = rng.uniform(0.01, 0.05, 2 * L)
    N = 10000
    samples = np.empty((N, 2 * L))
    for s in range(N):
        samples[s] = _qdc_odr(chi_meas + chi_std * rng.normal(size=2 * L), chi_cal + cal_std * rng.normal(size=2 * L), chi_exact, L, 0.01)
    mc_std = np.nanstd(samples, axis=0)
    out, err = _call(odr_uncertainty, chi_meas, chi_std, chi_cal, cal_std, chi_exact, L, suppression_threshold=0.01, n_samples=2000, seed=0)
    if err:
        msgs.append(f"odr_uncertainty raised {err}")
    else:
        out = np.asarray(out, dtype=float)
        if out.shape != (2 * L,):
            msgs.append("odr_uncertainty must return 2L standard deviations")
        else:
            mask = (f > 0.2) & np.isfinite(mc_std) & (mc_std > 0)
            rel = np.abs(out[mask] - mc_std[mask]) / mc_std[mask]
            worst = float(np.max(rel)) if mask.any() else 0.0
            if worst < 0.25:
                pts += 2.0
            elif worst < 0.5:
                pts += 1.0; msgs.append(f"odr_uncertainty within 50 % but not 25 % of the Monte-Carlo (worst {worst:.2f})")
            else:
                msgs.append(f"odr_uncertainty deviates by up to {worst * 100:.0f} % from a 10^4-sample Monte-Carlo (resample chi AND chi_cal with their stds, push each sample through your ODR, take the std per site)")
    # bias
    rng = np.random.default_rng(3300)
    ct = rng.uniform(0, 2, 12); fp = rng.uniform(0.1, 0.9, 12); fc = rng.uniform(0.1, 0.9, 12)
    out, err = _call(odr_bias, ct, fp, fc)
    exp = (1 - ct) * (1 - fp / fc)
    ok = err is None and np.asarray(out, dtype=float).shape == exp.shape and np.max(np.abs(np.asarray(out, float) - exp)) < 1e-12
    if ok:
        outs, errs_ = _call(odr_bias, 0.3, 0.5, 0.6)
        ok = errs_ is None and abs(float(outs) - (1 - 0.3) * (1 - 0.5 / 0.6)) < 1e-12
    if ok:
        pts += 2.0
    else:
        msgs.append(f"odr_bias(chi_true, f_phys, f_cal) must equal (1 - chi_true) (1 - f_phys/f_cal) for arrays and scalars ({err or 'value mismatch'})")
    detail = "ODR exact on synthetic CP-symmetric data (L=6,8,34), uncertainty within 25 % of MC, bias formula exact" if pts == 7 else "; ".join(msgs)
    return _record("ex3.1", pts, detail)


def _active_reduce(qc, active: list[int]):
    """Rebuild a circuit on the active physical qubits only (returns None if a gate touches others)."""
    from qiskit import QuantumCircuit
    pos = {q: i for i, q in enumerate(active)}
    out = QuantumCircuit(len(active))
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure", "delay") or getattr(inst.operation, "_directive", False):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if any(q not in pos for q in qs):
            return None
        out.append(inst.operation, [pos[q] for q in qs])
    return out


def _blocks_before_cz(qc):
    """For each 2q gate (in order) the 1q-gate unitaries on its two qubits since the previous 2q gate."""
    from qiskit.quantum_info import Operator
    pending: dict[int, np.ndarray] = {}
    blocks = []
    for inst in qc.data:
        if getattr(inst.operation, "_directive", False) or inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) == 1:
            U = Operator(inst.operation).data
            pending[qs[0]] = U @ pending.get(qs[0], np.eye(2))
        elif len(qs) == 2:
            blocks.append((tuple(qs), pending.pop(qs[0], np.eye(2)), pending.pop(qs[1], np.eye(2))))
    return blocks


def grade_ex3_2(twirl_circuit, postselect_charge) -> float:
    """twirl_circuit(isa_circuit, seed) on the grader's own L=6, t=2 FakeKingston ISA circuit: (1) ISA-valid with the
    same ordered CZ list; (1.5) unitary-equivalent on the 12 active qubits for 4 seeds; (1.5) >= 70 % of the CZs get a
    non-identity Pauli frame and 8 seeds give 8 distinct circuits; (1) postselect_charge(counts, L) keeps only
    Hamming-weight-L bitstrings (counts keys are Qiskit little-endian strings; the weight is order-independent)."""
    _require_callable("twirl_circuit", twirl_circuit); _require_callable("postselect_charge", postselect_charge)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    r = _refs(); backend = FakeKingston()
    base = _qpy_load(r["isa_circuit_L6_t2_kingston_O1_qpy"])
    active = [int(q) for q in r["isa_circuit_L6_t2_kingston_O1_layout"]]
    base_pairs = _cz_pairs(base)
    base_red = _active_reduce(base, active)
    rng = np.random.default_rng(3202)
    psis = []
    for _ in range(2):
        v = rng.normal(size=2 ** 12) + 1j * rng.normal(size=2 ** 12); psis.append(v / np.linalg.norm(v))
    base_out = [Statevector(p).evolve(base_red).data for p in psis]
    base_blocks = _blocks_before_cz(base)
    pts = 0.0; msgs = []
    seeds = list(range(8))
    circs = {}
    struct_ok = True; equiv_ok = True; frames = []; equiv_checked = 0
    for s in seeds:
        tc, err = _call(twirl_circuit, base.copy(), s)
        if err or not isinstance(tc, QuantumCircuit):
            msgs.append(f"twirl_circuit(seed={s}) -> {err or type(tc).__name__}"); struct_ok = False; equiv_ok = False; break
        circs[s] = tc
        ok, why = _is_isa(tc, backend)
        if not ok:
            struct_ok = False; msgs.append(f"seed {s}: {why} (express Paulis with x / rz(pi) / sx-based gates or re-run the translation pass)"); break
        if _cz_pairs(tc) != base_pairs:
            struct_ok = False; msgs.append(f"seed {s}: the ordered list of CZ pairs changed (twirling must only add single-qubit frames around each CZ)"); break
        red = _active_reduce(tc, active)
        if red is None:
            struct_ok = False; msgs.append(f"seed {s}: gates were added on qubits outside the active chain"); break
        if s < 4:
            equiv_checked += 1
            for p, bo in zip(psis, base_out):
                e = _phase_aligned_maxdiff(Statevector(p).evolve(red).data, bo)
                if e > 1e-8:
                    equiv_ok = False; msgs.append(f"seed {s}: twirled circuit is not equivalent to the original (max dev {e:.1e}); undo each Pauli after the CZ with CZ P CZ (X->XZ, Y->YZ, Z->Z on the partner)"); break
            if not equiv_ok:
                break
        blocks = _blocks_before_cz(tc)
        if len(blocks) == len(base_blocks):
            nd = 0
            for (q, a, b), (q0, a0, b0) in zip(blocks, base_blocks):
                if not _operator_close(a, a0, 1e-8)[0] or not _operator_close(b, b0, 1e-8)[0]:
                    nd += 1
            frames.append(nd / max(len(blocks), 1))
    if struct_ok:
        pts += 1.0
    if struct_ok and equiv_ok and equiv_checked == 4:
        pts += 1.5
    if struct_ok and len(circs) == 8:
        distinct = len({tuple((i.operation.name, tuple(np.round(np.asarray(i.operation.params, dtype=float), 8)), tuple(c.find_bit(q).index for q in i.qubits)) for i in c.data) for c in circs.values()}) == 8
        frac = float(np.mean(frames)) if frames else 0.0
        if distinct and frac >= 0.7:
            pts += 1.5
        else:
            msgs.append(f"{frac * 100:.0f} % of the CZs carry a non-identity frame (need >= 70 %) and {'8' if distinct else '< 8'} distinct circuits for 8 seeds (draw a random 2-qubit Pauli per CZ from the seed)")
    # postselection
    L = 4; rng = np.random.default_rng(3203)
    counts = {}
    for _ in range(40):
        bits = "".join(rng.choice(["0", "1"], size=2 * L)); counts[bits] = counts.get(bits, 0) + int(rng.integers(1, 50))
    counts["10101010"] = 77; counts["11110000"] = 33; counts["11111111"] = 5; counts["00000000"] = 6
    keep = {k: v for k, v in counts.items() if k.count("1") == L}
    out, err = _call(postselect_charge, dict(counts), L)
    ok = False
    if err is None:
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict) and all(isinstance(k, str) for k in out):
            ok = {k: int(v) for k, v in out.items() if v} == keep
    if ok:
        pts += 1.0
    else:
        msgs.append(f"postselect_charge: {err or 'must return {bitstring: count} keeping exactly the bitstrings with Hamming weight L (Q_tot = n_1 - L = 0)'}")
    detail = f"twirled circuits ISA + equivalent, {np.mean(frames) * 100:.0f} % dressed CZs, 8 distinct seeds; charge post-selection ok" if pts == 5 else "; ".join(msgs)
    return _record("ex3.2", pts, detail, extra={"frame_fraction": frames})


def _noise_model_params(noise_model) -> dict:
    """Parse NoiseModel.to_dict(): {('cz', (a,b)): p_depol, ('sx', (q,)): p, ('ro', (q,)): p}."""
    d = noise_model.to_dict()
    out = {}
    for e in d.get("errors", []):
        gq = [tuple(int(x) for x in q) for q in e.get("gate_qubits", [])]
        if e["type"] == "roerror":
            P = np.asarray(e["probabilities"], float)
            for q in gq:
                out[("ro", q)] = float(0.5 * (P[0, 1] + P[1, 0]))
                out[("ro_asym", q)] = float(abs(P[0, 1] - P[1, 0]))
        elif e["type"] == "qerror":
            probs = np.asarray(e["probabilities"], float)
            ins = e["instructions"]
            nq = len(gq[0]) if gq else 1
            p_id = 0.0
            for pr, circ in zip(probs, ins):
                if all(c["name"] == "id" or (c["name"] == "pauli" and set(c["params"][0]) == {"I"}) for c in circ):
                    p_id += pr
            dim = 4 ** nq
            p_depol = (1 - p_id) * dim / (dim - 1)
            for op in e["operations"]:
                for q in gq:
                    out[(op, q)] = float(p_depol)
    return out


def grade_ex3_3(noise_model, chain, chi_raw_arrays, X_raw, X_mit, odr_mitigate, backend) -> float:
    """Noisy rehearsal at L=6, t=4.  (1) chain: 12 distinct physical qubits forming a path; (2) noise_model
    parameters equal the frozen target values on that chain (CZ depolarizing = CZ error, sx depolarizing = sx error,
    readout flip = readout error; 1e-9); (2) X_raw = chi_wave - chi_vacuum and X_mit reproduced with the participant's
    odr_mitigate and the t=0 exact arrays (1e-9); (2) RMSE(X_mit) <= 0.8 RMSE(X_raw) and <= 0.10 vs the noiseless
    circuit; (1) ODR factors in (0.02, 0.97) on >= 8 of 12 sites."""
    _require_callable("odr_mitigate", odr_mitigate); _require_type("chi_raw_arrays", chi_raw_arrays, dict)
    if not hasattr(noise_model, "to_dict"):
        raise TypeError("noise_model must be a qiskit_aer.noise.NoiseModel")
    r = _refs(); L = 6; pts = 0.0; msgs = []
    chain = [int(q) for q in chain]
    summ = target_summary(backend)
    ok, why = chain_validity(summ, chain, 2 * L)
    if ok:
        pts += 1.0
    else:
        msgs.append(f"chain: {why}")
    # noise model parameters
    params = _noise_model_params(noise_model)
    edges = list(zip(chain[:-1], chain[1:]))
    bad = []
    for a, b in edges:
        e = _edge_error(summ, a, b)
        p = params.get(("cz", (a, b))); p2 = params.get(("cz", (b, a)))
        vals = [v for v in (p, p2) if v is not None]
        if not vals:
            bad.append(f"no cz error on ({a},{b})")
        elif any(abs(v - e) > 1e-9 for v in vals):
            bad.append(f"cz ({a},{b}): depolarizing {vals[0]:.3e} != target {e:.3e}")
    for q in chain:
        sx = float(summ["sx_error"][q]); ro = float(summ["readout_error"][q])
        p = params.get(("sx", (q,)))
        if p is None or abs(p - sx) > 1e-9:
            bad.append(f"sx q{q}: {p} != {sx:.3e}")
        pr = params.get(("ro", (q,)))
        if pr is None or abs(pr - ro) > 1e-9 or params.get(("ro_asym", (q,)), 0) > 1e-9:
            bad.append(f"readout q{q}: {pr} != {ro:.3e} (symmetric ReadoutError([[1-p,p],[p,1-p]]))")
    if not bad:
        pts += 2.0
    else:
        msgs.append("noise_model: " + "; ".join(bad[:3]) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""))
    # arrays
    need = ["chi_wave", "chi_wave_mitig", "chi_vacuum", "chi_vacuum_mitig"]
    if any(k not in chi_raw_arrays for k in need):
        raise TypeError(f"chi_raw_arrays must contain {need}")
    A = {k: np.asarray(chi_raw_arrays[k], dtype=float) for k in need}
    if any(A[k].shape != (2 * L,) for k in need):
        raise TypeError("chi_raw_arrays entries must have 12 values (L=6)")
    X_raw = np.asarray(X_raw, float); X_mit = np.asarray(X_mit, float)
    cw_ex = r["chi_wave_t0_L6"]
    cv_ex_opts = [r["chi_vacuum_t0_L6"], r["chi_vacuum_sub_t0_L6"]]
    if "chi_wave_exact" in chi_raw_arrays:
        cwe = np.asarray(chi_raw_arrays["chi_wave_exact"], float)
        if np.max(np.abs(cwe - cw_ex)) < 1e-3:
            cw_ex = cwe
        else:
            msgs.append("chi_wave_exact deviates from the exact t=0 wavepacket condensate")
    if "chi_vacuum_exact" in chi_raw_arrays:
        cve = np.asarray(chi_raw_arrays["chi_vacuum_exact"], float)
        if min(np.max(np.abs(cve - c)) for c in cv_ex_opts) < 1e-3:
            cv_ex_opts = [cve]
        else:
            msgs.append("chi_vacuum_exact deviates from the exact t=0 vacuum condensate")
    rec_ok = False; xm = None
    if np.max(np.abs(X_raw - (A["chi_wave"] - A["chi_vacuum"]))) < 1e-9:
        for cv_ex in cv_ex_opts:
            mw, e1 = _call(odr_mitigate, A["chi_wave"], A["chi_wave_mitig"], cw_ex, L)
            mv, e2 = _call(odr_mitigate, A["chi_vacuum"], A["chi_vacuum_mitig"], cv_ex, L)
            if e1 or e2:
                msgs.append(f"odr_mitigate raised {e1 or e2}"); break
            xm = np.asarray(mw, float) - np.asarray(mv, float)
            if np.allclose(xm, X_mit, atol=1e-9, equal_nan=True):
                rec_ok = True; break
        if not rec_ok and xm is not None:
            msgs.append("X_mit is not odr_mitigate(wave) - odr_mitigate(vacuum) computed from chi_raw_arrays with the exact t=0 condensates")
    else:
        msgs.append("X_raw != chi_wave - chi_vacuum")
    if rec_ok:
        pts += 2.0
    Xex = r["X_trotter_L6_t4"]
    rmse = lambda x: float(np.sqrt(np.nanmean((x - Xex) ** 2)))  # noqa: E731
    er, em = rmse(X_raw), rmse(X_mit)
    if em <= 0.8 * er and em <= 0.10 and rec_ok:
        pts += 2.0
    elif rec_ok:
        msgs.append(f"RMSE vs the noiseless circuit: raw {er:.4f}, mitigated {em:.4f} (need <= 0.8 x raw and <= 0.10)")
    f = (1 - A["chi_wave_mitig"]) / (1 - cw_ex)
    n_in = int(np.sum((f > 0.02) & (f < 0.97)))
    if n_in >= 8:
        pts += 1.0
    else:
        msgs.append(f"ODR suppression factors in (0.02, 0.97) on only {n_in}/12 sites: {np.round(f, 3)} (is the noise model attached to the simulator, and to the physical qubits of the chain?)")
    detail = f"reduced noise model matches the target on chain {chain}; RMSE raw {er:.4f} -> ODR {em:.4f}; factors {np.round(f, 2).tolist()}" if pts == 8 else "; ".join(msgs)
    return _record("ex3.3", pts, detail, arrays={**A, "X_raw": X_raw, "X_mit": X_mit, "chain": np.asarray(chain), "factors": f},
                   extra={"rmse_raw": er, "rmse_mit": em})


# ---------------------------------------------------------------------------
# Part 4 -- hardware
# ---------------------------------------------------------------------------
def _qdc_t0_exact():
    rd = os.path.join(ROOT, "reference_data")
    return (np.loadtxt(os.path.join(rd, "chi_wave_t0_sim_L34.txt")), np.loadtxt(os.path.join(rd, "chi_vacuum_t0_sim_L34.txt")))


def _arr(d: dict, *keys, n=68):
    for k in keys:
        if k in d and d[k] is not None:
            a = np.asarray(d[k], dtype=float).ravel()
            if a.shape == (n,):
                return a
    return None


def grade_ex4_1(canary_result, job_info) -> float:
    """t=0 canary: (1) job metadata (job_id, backend, usage_s <= 30, layout); (1) chi_wave / chi_vacuum arrays
    (68 values, finite, Pearson >= 0.8 with the exact t=0 profiles); (1) interpretation: 'flagged_sites' (list)
    and 'verdict' (str) present, flagged sites consistent with retention < 0.4."""
    _require_type("canary_result", canary_result, dict); _require_type("job_info", job_info, dict)
    pts = 0.0; msgs = []
    fallback = bool(canary_result.get("fallback", False) or job_info.get("fallback", False))
    usage = job_info.get("usage_s")
    meta_ok = all(k in job_info for k in ("job_id", "backend")) and isinstance(usage, (int, float)) and 0 <= usage <= 30
    lay = job_info.get("layout", job_info.get("chain"))
    if meta_ok and lay is not None and len(lay) == 68:
        pts += 1.0
    else:
        msgs.append(f"job_info needs job_id, backend, usage_s (<= 30 s; got {usage}) and the 68-qubit layout")
    cw = _arr(canary_result, "chi_wave", "wave", "chi_wave_t0"); cv = _arr(canary_result, "chi_vacuum", "vacuum", "chi_vacuum_t0")
    if cw is None or cv is None:
        ev = canary_result.get("evs")
        if ev is not None and np.asarray(ev, float).shape == (2, 68):
            cw, cv = np.asarray(ev, float)
    ew, evac = _qdc_t0_exact()
    ret = None
    if cw is None or cv is None:
        msgs.append("canary_result must contain 'chi_wave' and 'chi_vacuum' (68 values each, chi_j = (-1)^j <Z_j> + 1)")
    elif not (np.all(np.isfinite(cw)) and np.all(np.isfinite(cv))):
        msgs.append("canary arrays contain non-finite values")
    else:
        pw = np.corrcoef(cw, ew)[0, 1]
        packet = float(np.mean((cw - cv)[[33, 34]]) / np.mean((ew - evac)[[33, 34]]))
        if pw >= 0.5 and packet >= 0.05:
            pts += 1.0
        else:
            msgs.append(f"wave-arm Pearson with the exact t=0 profile {pw:.2f} (need >= 0.5) and centre packet retention {packet:.2f} (need >= 0.05): check the observable sign (-1)^j and the layout order")
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (1 - cv) / (1 - evac)
    flagged = canary_result.get("flagged_sites", canary_result.get("flagged_qubits"))
    verdict = canary_result.get("verdict")
    if isinstance(flagged, (list, tuple, np.ndarray)) and isinstance(verdict, str) and ret is not None:
        exp_flag = set(int(j) for j in np.where(ret < 0.4)[0])
        got = set(int(j) for j in flagged)
        if len(exp_flag ^ got) <= 2 or (lay is not None and len(set(int(lay[j]) for j in exp_flag) ^ got) <= 2):
            pts += 1.0
        else:
            msgs.append(f"flagged sites {sorted(got)} do not match the sites with vacuum-arm retention (1-chi)/(1-chi_exact) < 0.4: {sorted(exp_flag)}")
    else:
        msgs.append("canary_result needs 'flagged_sites' (list of sites with retention < 0.4) and a 'verdict' string")
    if fallback:
        pts = min(pts, 0.75 * MAX_POINTS["ex4.1"]); msgs.append("fallback dataset: capped at 75 %")
    detail = f"canary ok: usage {usage} s, median vacuum-arm retention {np.nanmedian(ret):.2f}, {len(flagged)} flagged sites, verdict '{verdict}'" if pts == 3 else "; ".join(msgs)
    return _record("ex4.1", pts, detail, arrays={"chi_wave": cw if cw is not None else np.array([]), "chi_vacuum": cv if cv is not None else np.array([])},
                   extra={"job_info": job_info, "fallback": fallback})


def hardware_metric(X_hat: np.ndarray, sigma: np.ndarray | None, X_ref: np.ndarray | None = None) -> dict:
    """SPEC E hardware metric (ex 4.2).  Returns the components and points before gates/caps."""
    if X_ref is None:
        X_ref = x_ref_t8(64)
    X_hat = np.asarray(X_hat, float); W = np.arange(25, 43)
    sig = np.asarray(sigma, float) if sigma is not None else np.zeros(68)
    xw = X_hat[W]; rw = X_ref[W]
    nan_w = int(np.sum(~np.isfinite(xw)))
    m = np.isfinite(xw)
    rmse_w = float(np.sqrt(np.mean((xw[m] - rw[m]) ** 2))) if m.any() else float("nan")
    peaks = [31, 32, 35, 36]; dips = [33, 34]
    C = float(np.nanmean(X_hat[peaks]) - np.nanmean(X_hat[dips]))
    C_ref = 0.515
    outside = np.array([j for j in range(68) if j not in set(W.tolist())])
    quiet = float(np.nanmedian(np.abs(X_hat[outside])))
    tol = 2 * np.sqrt(sig[W] ** 2 + 0.003 ** 2)
    cov = float(np.mean(np.abs(xw[m] - rw[m]) <= tol[m])) if m.any() else 0.0
    sep = float(np.nanmin(X_hat[peaks]) - np.nanmax(X_hat[dips]))
    clip = lambda x: float(min(max(x, 0.0), 1.0))  # noqa: E731
    p_rmse = 10 * clip(1 - rmse_w / 0.25) if np.isfinite(rmse_w) else 0.0
    p_rmse *= max(0.0, 1 - 0.05 * nan_w)
    p_contrast = 4 * clip(1 - abs(C - C_ref) / 0.35) if np.isfinite(C) else 0.0
    p_shape = 2.0 if (np.isfinite(C) and C >= 0.20 and sep >= 0.25) else 0.0
    p_quiet = 2.0 if quiet <= 0.05 else 0.0
    full = np.isfinite(X_hat)
    rmse_68 = float(np.sqrt(np.mean((X_hat[full] - X_ref[full]) ** 2))) if full.any() else float("nan")
    return {"rmse_w": rmse_w, "nan_in_window": nan_w, "contrast": C, "contrast_ref": C_ref, "separation": sep, "quiet": quiet,
            "coverage": cov, "rmse_68": rmse_68, "p_rmse": p_rmse, "p_contrast": p_contrast, "p_shape": p_shape, "p_quiet": p_quiet,
            "points_raw": p_rmse + p_contrast + p_shape + p_quiet}


def _hw_arrays(hardware_result: dict):
    keys = {"chi_wave": ("chi_wave", "wave"), "chi_wave_mitig": ("chi_wave_mitig", "wave_mitig"),
            "chi_vacuum": ("chi_vacuum", "vacuum"), "chi_vacuum_mitig": ("chi_vacuum_mitig", "vacuum_mitig")}
    A = {}
    for k, alts in keys.items():
        a = _arr(hardware_result, *alts)
        if a is None:
            raise TypeError(f"hardware_result must contain '{k}' (68 values)")
        A[k] = a
    S = {}
    for k in keys:
        S[k] = _arr(hardware_result, k + "_std")
    return A, S


def score_hardware(hardware_result: dict, job_info: dict, odr_mitigate: Callable, quiet_note_accepted: bool = False) -> dict:
    """Full ex 4.2 scoring (recompute X_hat, metric, gates, caps).  Shared with organizer/grade_submission.py."""
    A, S = _hw_arrays(hardware_result)
    ew, evac = _qdc_t0_exact()
    cw_ex = _arr(hardware_result, "chi_wave_exact"); cv_ex = _arr(hardware_result, "chi_vacuum_exact")
    if cw_ex is None or np.max(np.abs(cw_ex - ew)) > 1e-3:
        cw_ex = ew
    if cv_ex is None or np.max(np.abs(cv_ex - evac)) > 1e-3:
        cv_ex = evac
    L = 34
    mw, e1 = _call(odr_mitigate, A["chi_wave"], A["chi_wave_mitig"], cw_ex, L)
    mv, e2 = _call(odr_mitigate, A["chi_vacuum"], A["chi_vacuum_mitig"], cv_ex, L)
    out: dict[str, Any] = {"flags": [], "points": 0.0}
    if e1 or e2:
        out["flags"].append(f"odr_mitigate raised {e1 or e2}"); return out
    X_hat = np.asarray(mw, float) - np.asarray(mv, float)
    X_mit = _arr(hardware_result, "X_mit")
    if X_mit is None or not np.allclose(X_hat, X_mit, atol=1e-6, equal_nan=True):
        out["flags"].append("JUDGE FLAG: submitted X_mit is not reproduced by the team's odr_mitigate from the submitted evs (1e-6)")
        out["x_hat_consistent"] = False
        return out
    out["x_hat_consistent"] = True
    sigma = _arr(hardware_result, "sigma", "sigma_X", "X_mit_std")
    if sigma is None:
        # fallback: Monte-Carlo propagation of the Estimator stds through the QDC formula
        rng = np.random.default_rng(42)
        if all(S[k] is not None for k in S):
            samp = np.empty((300, 68))
            for i in range(300):
                samp[i] = (_qdc_odr(A["chi_wave"] + S["chi_wave"] * rng.normal(size=68), A["chi_wave_mitig"] + S["chi_wave_mitig"] * rng.normal(size=68), cw_ex, L)
                           - _qdc_odr(A["chi_vacuum"] + S["chi_vacuum"] * rng.normal(size=68), A["chi_vacuum_mitig"] + S["chi_vacuum_mitig"] * rng.normal(size=68), cv_ex, L))
            sigma = np.nanstd(samp, axis=0)
            out["flags"].append("sigma missing: grader propagated the Estimator stds (300 samples)")
        else:
            sigma = np.zeros(68)
    met = hardware_metric(X_hat, sigma)
    if quiet_note_accepted and met["p_quiet"] == 0:
        met["p_quiet"] = 2.0; met["points_raw"] += 2.0
    out.update(met); out["sigma"] = sigma; out["X_hat"] = X_hat
    pts = met["points_raw"]
    # gates
    factors = (1 - A["chi_wave_mitig"]) / (1 - cw_ex)
    n_f = int(np.sum(factors[25:43] < 0.9))
    out["odr_factors_below_0.9_in_window"] = n_f
    if n_f < 10:
        out["flags"].append(f"ODR factors < 0.9 on only {n_f}/18 window sites (gate: >= 10) -> 0 points")
        pts = 0.0
    lay = job_info.get("layout", job_info.get("chain"))
    if lay is None or len(lay) != 68:
        out["flags"].append("job_info lacks the 68-qubit layout shared by the 4 PUBs -> -2")
        pts -= 2.0
    usage = job_info.get("usage_s")
    fallback = bool(hardware_result.get("fallback", False) or job_info.get("fallback", False))
    if not isinstance(usage, (int, float)):
        out["flags"].append("job_info.usage_s missing -> -2"); pts -= 2.0
    elif usage > 180 and not fallback:
        out["flags"].append(f"usage {usage:.0f} s > 180 s -> -5"); pts -= 5.0
    elif usage > 180:
        # organizer-released data: the team spent no QPU time; the estimated usage of the organizer job is not gated
        out["flags"].append(f"usage {usage:.0f} s is the organizer's estimate for the fallback job (gate waived)")
    if fallback:
        pts = min(pts, 0.75 * MAX_POINTS["ex4.2"]); out["flags"].append("fallback dataset (organizer-released): capped at 75 %")
    out["fallback"] = fallback
    out["points"] = float(min(max(pts, 0.0), MAX_POINTS["ex4.2"]))
    out["usage_s"] = usage
    return out


def grade_ex4_2(hardware_result, job_info, odr_mitigate, quiet_note_accepted: bool = False) -> float:
    """Main t=8 run.  Metric (SPEC E): 10 clip(1 - RMSE_W/0.25) + 4 clip(1 - |C - 0.515|/0.35) + 2 [shape] + 2 [quiet <= 0.05];
    X_hat recomputed with the team's odr_mitigate must equal X_mit (1e-6); gates: ODR factors < 0.9 on >= 10 window
    sites, usage <= 180 s (-5; waived on organizer fallback data), fallback capped at 75 %."""
    _require_type("hardware_result", hardware_result, dict); _require_type("job_info", job_info, dict); _require_callable("odr_mitigate", odr_mitigate)
    res = score_hardware(hardware_result, job_info, odr_mitigate, quiet_note_accepted)
    pts = res["points"]
    if "rmse_w" in res:
        detail = (f"RMSE_W {res['rmse_w']:.3f} ({res['p_rmse']:.1f}), contrast {res['contrast']:.3f} ({res['p_contrast']:.1f}), shape {res['p_shape']:.0f}, "
                  f"quiet {res['quiet']:.3f} ({res['p_quiet']:.0f}), coverage {res['coverage']:.2f}, RMSE_68 {res['rmse_68']:.3f}, usage {res.get('usage_s')} s")
    else:
        detail = "not scored"
    if res["flags"]:
        detail += " | " + "; ".join(res["flags"])
    arrays = {k: v for k, v in hardware_result.items() if isinstance(v, (np.ndarray, list, tuple)) and np.asarray(v).dtype != object}
    arrays["X_hat"] = res.get("X_hat", np.array([])); arrays["sigma_used"] = res.get("sigma", np.array([]))
    extra = {k: v for k, v in res.items() if k not in ("X_hat", "sigma")}
    extra["job_info"] = job_info
    return _record("ex4.2", pts, detail, arrays=arrays, extra=extra)


def grade_ex4_3(improvement, odr_mitigate) -> float:
    """Improvement run (<= 60 s).  improvement = {'baseline': hardware_result-like, 'improved': hardware_result-like,
    'usage_s': float, 'rationale': str, 'z_score': float}.  (1) structure + usage <= 60 s (waived, and total capped at 75 %,
    when the runs are organizer fallback data / an illustration); (1) z-score of the RMSE_W
    change consistent with the grader's own (sigma from per-site uncertainties, 20 %); (1) rationale >= 150 chars naming
    the change; (1) real improvement (z >= 2 and lower RMSE_W) or 0.5 for an honest null result."""
    _require_type("improvement", improvement, dict); _require_callable("odr_mitigate", odr_mitigate)
    pts = 0.0; msgs = []
    need = ["baseline", "improved", "usage_s", "rationale", "z_score"]
    missing = [k for k in need if k not in improvement]
    if missing:
        return _record("ex4.3", 0.0, f"improvement dict missing {missing}")
    usage = improvement["usage_s"]
    fallback = bool(improvement.get("fallback", False) or improvement.get("illustration", False)
                    or any(isinstance(improvement.get(k), dict) and improvement[k].get("fallback", False) for k in ("baseline", "improved")))
    if isinstance(usage, (int, float)) and 0 <= usage <= 60:
        pts += 1.0
    elif fallback and isinstance(usage, (int, float)):
        pts += 1.0; msgs.append(f"usage_s = {usage} is the organizer's estimate for cached jobs (gate waived on fallback data)")
    else:
        msgs.append(f"usage_s = {usage} (must be <= 60 s)")
    X_ref = x_ref_t8(64); W = np.arange(25, 43)
    stats = {}
    for tag in ("baseline", "improved"):
        hr = improvement[tag]
        if not isinstance(hr, dict):
            return _record("ex4.3", pts, f"'{tag}' must be a hardware_result-like dict")
        res = score_hardware(hr, {"layout": [0] * 68, "usage_s": 0}, odr_mitigate)
        if "rmse_w" not in res:
            return _record("ex4.3", pts, f"'{tag}': {'; '.join(res['flags'])}")
        sig = np.asarray(res["sigma"], float)[W]; xw = res["X_hat"][W]; m = np.isfinite(xw)
        # sigma of RMSE_W by linear propagation: d RMSE / d X_j = (X_j - ref_j) / (n RMSE)
        n = m.sum(); rm = res["rmse_w"]
        grad = (xw[m] - X_ref[W][m]) / (n * rm) if rm > 0 else np.zeros(n)
        stats[tag] = (rm, float(np.sqrt(np.sum((grad * sig[m]) ** 2))))
    (rb, sb), (ri, si) = stats["baseline"], stats["improved"]
    z = (rb - ri) / math.sqrt(sb ** 2 + si ** 2) if (sb ** 2 + si ** 2) > 0 else float("inf")
    try:
        zs = float(improvement["z_score"])
    except (TypeError, ValueError):
        zs = float("nan")
    if np.isfinite(zs) and np.isfinite(z) and abs(zs - z) <= 0.2 * max(abs(z), 1.0):
        pts += 1.0
    else:
        msgs.append(f"z_score = {zs} but the grader gets z = (RMSE_base - RMSE_imp)/sqrt(s_b^2 + s_i^2) = {z:.2f} (RMSE_W {rb:.3f} +- {sb:.3f} -> {ri:.3f} +- {si:.3f})")
    rat = improvement["rationale"]
    if isinstance(rat, str) and len(rat.strip()) >= 150:
        pts += 1.0
    else:
        msgs.append("rationale must be a >= 150-character description of what was changed and why")
    if z >= 2 and ri < rb:
        pts += 1.0
    elif isinstance(rat, str) and len(rat.strip()) >= 150:
        pts += 0.5; msgs.append(f"no significant improvement (z = {z:.2f}); half credit for a documented null result")
    if fallback:
        pts = min(pts, 0.75 * MAX_POINTS["ex4.3"]); msgs.append("fallback / illustration on organizer data: capped at 75 %")
    detail = f"RMSE_W {rb:.3f} -> {ri:.3f}, z = {z:.2f}, usage {usage} s" + (" | " + "; ".join(msgs) if msgs else "")
    return _record("ex4.3", pts, detail, extra={"z_grader": z, "rmse_baseline": rb, "rmse_improved": ri, "usage_s": usage, "fallback": fallback})


# ---------------------------------------------------------------------------
# Bonus
# ---------------------------------------------------------------------------
def grade_bonus_B1(qc_dt05, qc_dt025, X_dt05, X_dt025, X_richardson) -> float:
    """L=34 dt -> 0 Richardson (bond 32).  (1) qc_dt05 has 16 Trotter steps (2q count within 1 % of the reference);
    (1) qc_dt025 has 32 steps; (1) X_richardson == (4 X_dt025 - X_dt05)/3 with finite 68-element profiles."""
    from qiskit import QuantumCircuit
    _require_type("qc_dt05", qc_dt05, QuantumCircuit); _require_type("qc_dt025", qc_dt025, QuantumCircuit)
    r = _refs(); pts = 0.0; msgs = []
    for qc, key, steps in ((qc_dt05, "n2q_logical_16steps", 16), (qc_dt025, "n2q_logical_32steps", 32)):
        n = _n2q(qc); ref = int(r[key])
        if abs(n - ref) <= 0.01 * ref:
            pts += 1.0
        else:
            msgs.append(f"{steps}-step circuit has {n} 2q gates, expected {ref} +-1 %")
    X05 = np.asarray(X_dt05, float); X025 = np.asarray(X_dt025, float); XR = np.asarray(X_richardson, float)
    if X05.shape == X025.shape == XR.shape == (68,) and np.all(np.isfinite(XR)) and np.max(np.abs(XR - (4 * X025 - X05) / 3)) < 1e-9:
        pts += 1.0
    else:
        msgs.append("X_richardson must equal (4 X_dt025 - X_dt05)/3 (68 finite values)")
    detail = "16/32-step circuits and Richardson profile ok (organizer compares the arrays offline)" if pts == 3 else "; ".join(msgs)
    return _record("B1", pts, detail, arrays={"X_dt05": X05, "X_dt025": X025, "X_richardson": XR})


def grade_bonus_B2(toy_results) -> float:
    """ODR on a depolarizing vs amplitude-damping toy at L=4 with/without Pauli twirling.  toy_results = {
    'raw_depol', 'odr_depol', 'raw_amp', 'odr_amp', 'odr_amp_twirl': max|X - X_exact|}.  (1) ODR removes the
    depolarizing error (odr_depol < 0.25 raw_depol) but not the amplitude-damping one (odr_amp > odr_depol);
    (1) witness ordering: twirling makes amplitude damping ODR-correctable again (odr_amp_twirl < odr_amp)."""
    _require_type("toy_results", toy_results, dict)
    need = ["raw_depol", "odr_depol", "raw_amp", "odr_amp", "odr_amp_twirl"]
    missing = [k for k in need if k not in toy_results]
    if missing:
        return _record("B2", 0.0, f"toy_results missing {missing}")
    v = {k: float(toy_results[k]) for k in need}
    pts = 0.0; msgs = []
    if v["odr_depol"] < 0.25 * v["raw_depol"] and v["odr_amp"] > v["odr_depol"]:
        pts += 1.0
    else:
        msgs.append(f"expected odr_depol < 0.25 raw_depol and odr_amp > odr_depol, got {v}")
    if v["odr_amp_twirl"] < v["odr_amp"]:
        pts += 1.0
    else:
        msgs.append("witness ordering: twirled amplitude damping should be mitigated better than untwirled (odr_amp_twirl < odr_amp)")
    detail = f"toy ordering confirmed: {v}" if pts == 2 else "; ".join(msgs)
    return _record("B2", pts, detail, extra={"values": v})


def grade_bonus_B3(barbell_rzz) -> float:
    """barbell_rzz(a1..a6) -> circuit/gate built from rzz (fractional gate) whose Operator equals challenge_utils.barbell."""
    _require_callable("barbell_rzz", barbell_rzz)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    import challenge_utils as cu
    rng = np.random.default_rng(3303)
    a = rng.uniform(-1, 1, 6)
    g, err = _call(barbell_rzz, *a)
    if err:
        return _record("B3", 0.0, f"barbell_rzz raised {err}")
    qc = g if isinstance(g, QuantumCircuit) else None
    if qc is None:
        try:
            qc = QuantumCircuit(4); qc.append(g, range(4))
        except Exception as e:  # noqa: BLE001
            return _record("B3", 0.0, f"barbell_rzz must return a 4-qubit circuit or gate ({e})")
    ops = qc.decompose(reps=1).count_ops() if qc.count_ops().get("rzz", 0) == 0 else qc.count_ops()
    if ops.get("rzz", 0) == 0 or any(k in ops for k in ("cx", "cz", "ecr")):
        return _record("B3", 0.0, f"circuit must use rzz gates and no cx/cz (ops: {dict(ops)})")
    ok, e = _operator_close(Operator(qc).data, Operator(cu.barbell(*a)).data, 1e-8)
    detail = f"rzz barbell equals the CX barbell up to phase ({ops.get('rzz')} rzz gates)" if ok else f"unitary differs from challenge_utils.barbell (max dev {e:.2e})"
    return _record("B3", 1.0 if ok else 0.0, detail)


__all__ = [n for n in dir() if n.startswith("grade_") or n in ("check_env", "summary", "hardware_metric", "score_hardware", "target_summary",
                                                                "chain_cost", "chain_validity", "baseline_chain", "x_ref_t8")]
