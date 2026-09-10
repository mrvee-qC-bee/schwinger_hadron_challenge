#!/bin/zsh
cd /Users/vishalbajpe/Downloads/QiskitFallFest/schwinger-hadron-challenge
P=/Users/vishalbajpe/miniforge3/envs/qiskit-paper/bin/python
while ! grep -q DONE reference_data/make_mps_reference.log; do sleep 20; done
echo "MPS done at $(date)"
$P organizer/make_grader_refs.py > scratch/make_grader_refs_final.log 2>&1
echo "refs regenerated: $(grep -E 'wrote|RMSE_0|C_ref_measured|bd64_vs' scratch/make_grader_refs_final.log)"
MPLBACKEND=Agg $P tools/test_grader.py > scratch/test_grader_final.log 2>&1
echo "test exit $?"; tail -5 scratch/test_grader_final.log
$P organizer/grade_submission.py scratch/submission_test --team testteam > scratch/grade_submission_final.log 2>&1
$P organizer/grade_submission.py scratch/submission_test --team testteam --hidden > scratch/grade_submission_hidden.log 2>&1
echo "ALL DONE"
