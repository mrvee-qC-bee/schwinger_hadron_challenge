"""
Organizer tool: regenerate reference_data/grader_refs.npz from organizer/schwinger_reference.py.

    python organizer/make_grader_refs.py            # ~2-4 min on a laptop

Everything the local autograder (fallfest_grader.py) compares against lives in this one npz so
that the participant kit never needs the organizer module.  Every stored number is printed.

Contents (all keys are plain numpy arrays; scalars stored as 0-d arrays):
  sv_vacuum_L{6,8}, sv_wave_L{6,8}         statevectors of prep_vacuum / prep_wave (default angles)
  sv_physics_t2_L6                          t=2 physics circuit (2 pinned Fig.-8 Trotter steps, dt=1) on prep_wave(6)
  gs_L{4,6,8}, E0_L{4,6,8}                  exact ground state / energy of the FULL H (with +m/2*I per site)
  E_adapt_L{4,6,8}, vqe_fid_L{4,6,8}        energy / fidelity of the 2-step SC-ADAPT-VQE vacuum
  H_full_labels_L{4,6,8}, H_full_coeffs_L{..}, H_trunc_labels_L{..}, H_trunc_coeffs_L{..}
  Hel_trunc_labels_L{4,6}, Hel_trunc_coeffs_L{4,6}   H_el^{(Q=0)}(1) alone (ex 1.2 electric layer)
  chi_wave_t0_L{6,8}, chi_vacuum_t0_L{6,8}   exact t=0 condensates
  X_exact_full_L8_t4, X_exact_trunc_L8_t4, truncation_shift_L8_t4 (= max_j |X_full - X_trunc|)
  X_exact_trunc_L8_t2, X_exact_trunc_L6_t4, X_exact_full_L6_t4
  X_trotter_L6_t4 (noiseless 4-step circuit, dt=1: the ex 3.3 target), chi_wave_trotter_L6_t4, chi_vacuum_trotter_L6_t4
  trotter_table_t{2,4}_dt{1,0.5,0.25}       max_j |X_trotter - X_exact_trunc| at L=8
  richardson_error_t4 (dt 0.5 & 0.25), richardson_error_t2
  window_lo, window_hi, C_ref, RMSE_0 (if mps_reference_L34.npz exists), X_ref_t8_bd64 (idem)
  tgt_<name>_edges (E,2) int, tgt_<name>_cz_error (E,), tgt_<name>_sx_error, tgt_<name>_x_error,
  tgt_<name>_readout_error, tgt_<name>_t1, tgt_<name>_t2 (nan where None), tgt_<name>_num_qubits
      for name in kingston, fez, marrakesh (frozen fake-backend targets, runtime 0.49.0)
  n2q_logical_t8, n_cz_physics_t8_kingston_O3_seed42, n2q_logical_16steps, n2q_logical_32steps
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import schwinger_reference as R  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector, SparsePauliOp  # noqa: E402
from scipy.sparse.linalg import eigsh, expm_multiply  # noqa: E402

OUT = os.path.join(ROOT, "reference_data", "grader_refs.npz")
store: dict[str, np.ndarray] = {}
T0 = time.time()


def put(key, val):
    arr = np.asarray(val)
    store[key] = arr
    if arr.ndim == 0:
        print(f"  {key} = {arr}")
    elif arr.size <= 20:
        print(f"  {key} = {np.array2string(arr, precision=6, max_line_width=200)}")
    else:
        print(f"  {key}: shape {arr.shape} dtype {arr.dtype} (first 4: {np.array2string(arr.ravel()[:4], precision=6)})")


def chi_from_state(psi: np.ndarray, L: int) -> np.ndarray:
    sv = Statevector(psi)
    return np.array([np.real(sv.expectation_value(o)) for o in R.chiral_condensate_observables(L)])


def exact_evolve(psi0: np.ndarray, H_sparse, t: float) -> np.ndarray:
    return expm_multiply(-1j * t * H_sparse, psi0)


def pauli_arrays(H: SparsePauliOp):
    H = H.simplify()
    labels = np.array([str(p) for p in H.paulis])
    coeffs = np.real_if_close(H.coeffs)
    return labels, np.asarray(coeffs, dtype=complex)


# ---------------------------------------------------------------------------
print("== 1. statevectors (prep_vacuum / prep_wave at L=6,8; t=2 physics circuit at L=6)")
for L in (6, 8):
    put(f"sv_vacuum_L{L}", Statevector(R.prep_vacuum(L)).data)
    put(f"sv_wave_L{L}", Statevector(R.prep_wave(L)).data)
    put(f"chi_wave_t0_L{L}", chi_from_state(store[f"sv_wave_L{L}"], L))
    put(f"chi_vacuum_t0_L{L}", chi_from_state(store[f"sv_vacuum_L{L}"], L))
qp, qm = R.evolve_circuits(R.prep_wave(6), 6, 2.0)
put("sv_physics_t2_L6", Statevector(qp).data)
put("mitig_return_fidelity_t2_L6", abs(np.vdot(store["sv_wave_L6"], Statevector(qm).data)) ** 2)
put("chi_vacuum_sub_t0_L6", chi_from_state(Statevector(R.prep_vacuum_for_subtraction(6)).data, 6))

# ---------------------------------------------------------------------------
print("== 2. exact ground states / energies (FULL H, +m/2*I convention) and ADAPT-VQE quality")
for L in (4, 6, 8):
    H = R.schwinger_hamiltonian(L)
    Hs = H.to_matrix(sparse=True)
    w, v = eigsh(Hs, k=1, which="SA", tol=1e-12)
    gs = v[:, 0]
    put(f"E0_L{L}", float(w[0]))
    put(f"gs_L{L}", gs.astype(complex))
    sv = Statevector(R.prep_vacuum(L))
    put(f"E_adapt_L{L}", float(np.real(sv.expectation_value(H))))
    put(f"vqe_fid_L{L}", float(abs(np.vdot(gs, sv.data)) ** 2))

# ---------------------------------------------------------------------------
print("== 3. Hamiltonian Pauli dictionaries (full & truncated) at L=4,6,8")
for L in (4, 6, 8):
    lab, co = pauli_arrays(R.schwinger_hamiltonian(L))
    put(f"H_full_labels_L{L}", lab); put(f"H_full_coeffs_L{L}", co)
    lab, co = pauli_arrays(R.schwinger_hamiltonian(L, truncated=True))
    put(f"H_trunc_labels_L{L}", lab); put(f"H_trunc_coeffs_L{L}", co)
    if L in (4, 6):
        lab, co = pauli_arrays(R.electric_hamiltonian_truncated(L))
        put(f"Hel_trunc_labels_L{L}", lab); put(f"Hel_trunc_coeffs_L{L}", co)

# ---------------------------------------------------------------------------
print("== 4. exact evolution: truncation shift (L=8, t=4) and exact X profiles")
for L, ts in ((8, (2.0, 4.0)), (6, (4.0,))):
    Hf = R.schwinger_hamiltonian(L).to_matrix(sparse=True)
    Ht = R.schwinger_hamiltonian(L, truncated=True).to_matrix(sparse=True)
    psi_w = store[f"sv_wave_L{L}"]
    psi_v = store[f"sv_vacuum_L{L}"]
    for t in ts:
        tt = int(t)
        Xt = chi_from_state(exact_evolve(psi_w, Ht, t), L) - chi_from_state(exact_evolve(psi_v, Ht, t), L)
        put(f"X_exact_trunc_L{L}_t{tt}", Xt)
        if (L, tt) in ((8, 4), (6, 4)):
            Xf = chi_from_state(exact_evolve(psi_w, Hf, t), L) - chi_from_state(exact_evolve(psi_v, Hf, t), L)
            put(f"X_exact_full_L{L}_t{tt}", Xf)
            put(f"truncation_shift_array_L{L}_t{tt}", Xf - Xt)
            put(f"truncation_shift_L{L}_t{tt}", float(np.max(np.abs(Xf - Xt))))

# ---------------------------------------------------------------------------
print("== 5. Trotter table at L=8 (max_j |X_trotter - X_exact_trunc|), ratios, Richardson")
L = 8
Xtr = {}
for t in (2.0, 4.0):
    for dt in (1.0, 0.5, 0.25):
        n_steps = int(round(t / dt))
        qw = R.prep_wave(L); qv = R.prep_vacuum(L)
        for _ in range(n_steps):
            qw = R.trotter_step(qw, L, dt); qv = R.trotter_step(qv, L, dt)
        X = chi_from_state(Statevector(qw).data, L) - chi_from_state(Statevector(qv).data, L)
        Xtr[(t, dt)] = X
        err = float(np.max(np.abs(X - store[f"X_exact_trunc_L8_t{int(t)}"])))
        put(f"trotter_table_t{int(t)}_dt{dt:g}", err)
        put(f"X_trotter_L8_t{int(t)}_dt{dt:g}", X)
    e1, e2, e3 = (store[f"trotter_table_t{int(t)}_dt{dt:g}"] for dt in (1.0, 0.5, 0.25))
    put(f"trotter_ratio_t{int(t)}_1_05", float(e1 / e2))
    put(f"trotter_ratio_t{int(t)}_05_025", float(e2 / e3))
    XR = (4 * Xtr[(t, 0.25)] - Xtr[(t, 0.5)]) / 3
    put(f"richardson_error_t{int(t)}", float(np.max(np.abs(XR - store[f"X_exact_trunc_L8_t{int(t)}"]))))
put("dominant_error", "trotter" if store["trotter_table_t4_dt1"] > store["truncation_shift_L8_t4"] else "truncation")

print("== 5b. noiseless Trotter circuit at L=6, t=4 (4 steps, dt=1): the ex 3.3 target")
L = 6
qw, _ = R.evolve_circuits(R.prep_wave(L), L, 4.0)
qv, _ = R.evolve_circuits(R.prep_vacuum(L), L, 4.0)
cw = chi_from_state(Statevector(qw).data, L); cv = chi_from_state(Statevector(qv).data, L)
put("chi_wave_trotter_L6_t4", cw); put("chi_vacuum_trotter_L6_t4", cv); put("X_trotter_L6_t4", cw - cv)
put("trotter_vs_exact_L6_t4", float(np.max(np.abs(cw - cv - store["X_exact_trunc_L6_t4"]))))

# ---------------------------------------------------------------------------
print("== 6. window constants / hardware reference")
put("window_lo", 25); put("window_hi", 42); put("C_ref", 0.515)
put("peak_sites", np.array([31, 32, 35, 36])); put("dip_sites", np.array([33, 34]))
mps_path = os.path.join(ROOT, "reference_data", "mps_reference_L34.npz")
if os.path.exists(mps_path):
    d = np.load(mps_path)
    Xref = d["chi_wave_t8_bd64"] - d["chi_vacuum_t8_bd64"]
    put("X_ref_t8_bd64", Xref)
    W = np.arange(25, 43)
    put("RMSE_0", float(np.sqrt(np.mean(Xref[W] ** 2))))
    put("C_ref_measured", float(np.mean(Xref[[31, 32, 35, 36]]) - np.mean(Xref[[33, 34]])))
    if "chi_wave_t8_bd128" in d.files:
        X128 = d["chi_wave_t8_bd128"] - d["chi_vacuum_t8_bd128"]
        put("X_ref_t8_bd128", X128)
        put("bd64_vs_bd128_max", float(np.max(np.abs(Xref - X128))))
else:
    print("  (mps_reference_L34.npz not present yet -- X_ref/RMSE_0 not stored; rerun later)")

# ---------------------------------------------------------------------------
print("== 7. frozen fake-backend target summaries")
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh  # noqa: E402


def target_summary(backend):
    tgt = backend.target
    nq = tgt.num_qubits
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    cz = []
    for (a, b) in edges:
        p = tgt["cz"].get((a, b)) or tgt["cz"].get((b, a))
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
    return {"num_qubits": nq, "edges": np.array(edges, dtype=int), "cz_error": np.array(cz),
            "sx_error": q1("sx"), "x_error": q1("x"), "readout_error": q1("measure"),
            "t1": np.array(t1), "t2": np.array(t2)}


for name, cls in (("kingston", FakeKingston), ("fez", FakeFez), ("marrakesh", FakeMarrakesh)):
    b = cls()
    s = target_summary(b)
    for k, v in s.items():
        put(f"tgt_{name}_{k}", v)
    print(f"  {b.name}: {len(s['edges'])} edges, CZ err median {np.nanmedian(s['cz_error']):.4f}, "
          f"#CZ>=0.5: {int(np.sum(s['cz_error'] >= 0.5))}, RO median {np.nanmedian(s['readout_error']):.4f}, "
          f"#RO>=0.5: {int(np.sum(s['readout_error'] >= 0.5))}, T1 None: {int(np.sum(np.isnan(s['t1'])))}")

# ---------------------------------------------------------------------------
print("== 8. gate accounting at L=34 (logical 2q counts; FakeKingston O3 seed 42 CZ count)")
L = 34
qw = R.prep_wave(L)
qp8, _ = R.evolve_circuits(qw, L, 8.0)
put("n2q_logical_t8", sum(1 for i in qp8.decompose(reps=3).data if len(i.qubits) > 1))
for ns in (16, 32):
    q, _ = R.evolve_circuits(qw, L, 8.0, n_steps=ns)
    put(f"n2q_logical_{ns}steps", sum(1 for i in q.decompose(reps=3).data if len(i.qubits) > 1))
if "--skip-transpile" not in sys.argv:
    from qiskit.transpiler import generate_preset_pass_manager
    t0 = time.time()
    kb = FakeKingston()
    pm = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42)
    isa = pm.run(qp8)
    put("n_cz_physics_t8_kingston_O3_seed42", isa.count_ops().get("cz", 0))
    put("depth2q_physics_t8_kingston_O3_seed42", R.two_qubit_depth(isa))
    put("layout_t8_kingston_O3_seed42", np.array(isa.layout.initial_index_layout(filter_ancillas=True)))
    isa1 = generate_preset_pass_manager(optimization_level=1, backend=kb, seed_transpiler=42).run(qp8)
    put("n_cz_physics_t8_kingston_O1_seed42", isa1.count_ops().get("cz", 0))
    print(f"  (transpile took {time.time() - t0:.1f}s)")

    print("== 9. grader-owned ISA circuit for ex 3.2 (L=6, t=2 physics circuit, FakeKingston O1 seed 42) as QPY bytes")
    import io
    from qiskit import qpy
    qp2, _ = R.evolve_circuits(R.prep_wave(6), 6, 2.0)
    isa2 = generate_preset_pass_manager(optimization_level=1, backend=kb, seed_transpiler=42).run(qp2)
    buf = io.BytesIO(); qpy.dump(isa2, buf)
    put("isa_circuit_L6_t2_kingston_O1_qpy", np.frombuffer(buf.getvalue(), dtype=np.uint8))
    put("isa_circuit_L6_t2_kingston_O1_layout", np.array(isa2.layout.initial_index_layout(filter_ancillas=True)))
    put("isa_circuit_L6_t2_kingston_O1_ncz", isa2.count_ops().get("cz", 0))
    put("sv_physics_t2_L6_check", float(abs(np.vdot(store["sv_physics_t2_L6"], Statevector(qp2).data)) ** 2))

np.savez(OUT, **store)
print(f"\nwrote {OUT} ({len(store)} keys, {os.path.getsize(OUT) / 1e6:.1f} MB) in {time.time() - T0:.1f}s")
