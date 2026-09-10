# Judge sheet — Part 5 (10 points)

Team: ______________________   Judge: ______________________   Slot: ________   Date: ________

Score each row independently; the three judges' sheets are averaged and rounded to 0.5.
The anchors are the ones in `ORGANIZER_GUIDE.md` §6.

| item | pts | 0 | half | full | score |
|---|---|---|---|---|---|
| Error-budget completeness | 3 | list of buzzwords | table with ≥ 4 of the 7 rows but no propagation to X | all 7 rows, each with a mechanism, a sign/shape and how it was measured or bounded | ___ / 3 |
| Magnitudes | 3 | none or wrong by > 10× | right order of magnitude for ≥ 4 rows | numbers consistent with their own notebook (Trotter dt = 1 error, MPS bond gap, shot noise from stds, retention factors, ODR bias) | ___ / 3 |
| Viva | 4 | cannot explain their own code | answers 4–6 of 8 questions | answers ≥ 7 of 8, including one "why" question | ___ / 4 |

The seven error-budget rows: (1) Trotter dt = 1, (2) electric-field range-1 truncation,
(3) SC-ADAPT-VQE vacuum infidelity, (4) MPS bond truncation of the reference, (5) shot noise +
twirl-sampling noise, (6) decoherence retention and its residual non-Pauli part, (7) ODR ratio bias
and the retention threshold / mirror averaging.

## Viva (7 minutes, 8 questions from the 20-question bank in §6; ★ recall, ★★ reasoning)

Pick at least three ★★ questions. The judges choose which team member answers each question.

| # | bank no. | ★/★★ | answered by | correct? (✓ / ~ / ✗) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |

## Flags from `organizer_score.json` to resolve at the viva

Copy every entry of the team's `flags` list (judge flags from the hardware metric, `TAMPER?` lines
from the tamper check, `NOT RE-VERIFIED` exercises) and record the outcome.

| flag | question asked | resolved? | action (none / zero exercise …) |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Exercises the team could not explain (a judge may zero any autograded part): ______________________

**Total Part 5: ___ / 10**   Signature: ______________________
