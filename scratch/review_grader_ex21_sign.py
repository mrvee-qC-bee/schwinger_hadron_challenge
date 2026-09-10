"""ex2.1 observable check: Pauli.__str__ truncates labels of > 50 qubits, so _pauli_dict keys collide on 156-qubit observables."""
from review_grader_common import setup, run
import numpy as np
ff = setup("ex21")
import schwinger_reference as R
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston
kb = FakeKingston()
p = Pauli("I" * 100 + "Z" + "I" * 55)
print("str(Pauli) on 156 qubits:", repr(str(p)), "len", len(str(p)), "| to_label len", len(p.to_label()))
L = 34
qw = R.prep_wave(L); qv = R.prep_vacuum_for_subtraction(L)
qpw, qmw = R.evolve_circuits(qw, L, 8.0, protect_midpoint=True); qpv, qmv = R.evolve_circuits(qv, L, 8.0, protect_midpoint=True)
pm = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42)
isa_w = pm.run(qpw); layout = isa_w.layout.initial_index_layout(filter_ancillas=True)
pm_l = generate_preset_pass_manager(optimization_level=3, backend=kb, seed_transpiler=42, initial_layout=layout)
circs = [isa_w] + [pm_l.run(c) for c in (qmw, qpv, qmv)]
obs = [o.apply_layout(isa_w.layout) for o in R.chiral_condensate_observables(L)]
print("layout:", layout)
print("_pauli_dict of the correct chi_20 (phys qubit %d):" % layout[20], ff._pauli_dict(obs[20], False))
# WRONG observables: sign of Z flipped on every site whose physical qubit is < 106 (Z beyond the 50-char label prefix)
wrong = []
for j, o in enumerate(obs):
    if layout[j] < 106:
        wrong.append(SparsePauliOp(o.paulis, [-c if str(p_).count("Z") else c for p_, c in zip(o.paulis, o.coeffs)]) if False else
                     SparsePauliOp.from_list([(o.paulis[k].to_label(), (-o.coeffs[k] if o.paulis[k].to_label().count("Z") else o.coeffs[k])) for k in range(len(o))]))
    else:
        wrong.append(o)
n_flipped = sum(1 for j in range(68) if layout[j] < 106)
print(f"flipped the Z sign on {n_flipped} of 68 observables (physical qubit < 106)")
run("ex2.1 wrong observables: Z sign flipped on sites with phys qubit < 106", ff.grade_ex2_1, circs, wrong, kb)
# observables on the WRONG qubits (shifted by one site) where invisible
wrong2 = [obs[(j + 1) % 68] if (layout[j] < 106 and layout[(j + 1) % 68] < 106) else obs[j] for j in range(68)]
run("ex2.1 wrong observables: chi_j replaced by chi_{j+1} where both invisible", ff.grade_ex2_1, circs, wrong2, kb)
# a plain wrong observable that is visible: sign flipped on site 0 (phys qubit 141)
wrong3 = list(obs); wrong3[0] = SparsePauliOp.from_list([(obs[0].paulis[k].to_label(), (-obs[0].coeffs[k] if obs[0].paulis[k].to_label().count("Z") else obs[0].coeffs[k])) for k in range(2)])
run("ex2.1 control: sign flipped on site 0 (phys 141, visible)", ff.grade_ex2_1, circs, wrong3, kb)
run("ex2.1 control: correct observables", ff.grade_ex2_1, circs, obs, kb)
